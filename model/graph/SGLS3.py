import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor

# Paper: self-supervised graph learning for recommendation. SIGIR'21


class SGLS3(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SGLS3, self).__init__(conf, training_set, test_set)
        args = self.config['SGLS3']

        self.cl_rate = float(args['lambda'])
        aug_type = self.aug_type = int(args['aug_type'])
        drop_rate = float(args['drop_rate'])
        n_layers = int(args['n_layer'])
        temp = float(args['temp'])

        # debiased negative sampling params
        self.debias_temp_start = float(args.get('debias_temp_start', 0.95))
        self.debias_temp_end = float(args.get('debias_temp_end', 0.6))
        self.debias_pool = int(args.get('debias_pool', 20))
        self.cooc_threshold = int(args.get('cooc_threshold', 3))

        # precompute item-user interaction sets
        self.item_user_set = {}
        for u in self.data.training_set_u:
            for i in self.data.training_set_u[u]:
                self.item_user_set.setdefault(i, set()).add(u)

        # precompute item popularity
        self.item_popularity = {
            i: len(self.item_user_set.get(i, []))
            for i in self.data.item
        }

        self.model = SGL_Encoder(
            self.data,
            self.emb_size,
            drop_rate,
            n_layers,
            temp,
            aug_type
        )

    def _current_debias_threshold(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.debias_temp_start + (self.debias_temp_end - self.debias_temp_start) * progress

    def _cooc_filter(self, pos_item, neg_item):
        pos_users = self.item_user_set.get(self.data.id2item[pos_item], set())
        neg_users = self.item_user_set.get(self.data.id2item[neg_item], set())
        return len(pos_users & neg_users) <= self.cooc_threshold

    def _remove_popular_batch(self, neg_list):
        if len(neg_list) <= 3:
            return neg_list

        pop_scores = [
            (i, self.item_popularity.get(self.data.id2item[i], 0))
            for i in neg_list
        ]
        pop_scores.sort(key=lambda x: x[1], reverse=True)

        remove_set = set([x[0] for x in pop_scores[:3]])
        filtered = [i for i in neg_list if i not in remove_set]

        return filtered if len(filtered) > 0 else neg_list

    def _debias_negatives(self, user_idx, pos_idx, neg_idx, rec_user_emb, rec_item_emb, epoch):
        threshold = self._current_debias_threshold(epoch)
        item_keys = list(self.data.item.keys())
        debiased_neg = []

        for u, pos, neg in zip(user_idx, pos_idx, neg_idx):
            prob = torch.sigmoid(torch.matmul(rec_user_emb[u], rec_item_emb[neg])).item()

            # co-occurrence filtering
            if not self._cooc_filter(pos, neg):
                prob = 1.0

            # keep current negative if acceptable
            if prob <= threshold:
                debiased_neg.append(neg)
                continue

            # otherwise resample from candidate pool
            user_name = self.data.id2user[u]
            pool = []

            while len(pool) < self.debias_pool:
                cand = random.choice(item_keys)
                if cand not in self.data.training_set_u[user_name]:
                    pool.append(self.data.item[cand])

            pool_scores = torch.sigmoid(torch.matmul(rec_item_emb[pool], rec_user_emb[u]))

            valid_mask = pool_scores <= threshold
            valid_candidates = []

            for idx in torch.nonzero(valid_mask).view(-1).tolist():
                candidate = pool[idx]
                if self._cooc_filter(pos, candidate):
                    valid_candidates.append(candidate)

            if len(valid_candidates) > 0:
                debiased_neg.append(random.choice(valid_candidates))
            else:
                best = pool[torch.argmin(pool_scores).item()]
                debiased_neg.append(best)

        # remove top-3 popular negatives in current batch
        debiased_neg = self._remove_popular_batch(debiased_neg)

        # pad if length mismatch
        while len(debiased_neg) < len(user_idx):
            debiased_neg.append(random.choice(debiased_neg))

        return debiased_neg

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()

                neg_idx = self._debias_negatives(
                    user_idx,
                    pos_idx,
                    neg_idx,
                    rec_user_emb.detach(),
                    rec_item_emb.detach(),
                    epoch
                )

                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss(
                    [user_idx, pos_idx],
                    dropped_adj1,
                    dropped_adj2
                )

                batch_loss = (
                    rec_loss
                    + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                    + cl_loss
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'rec_loss:', rec_loss.item(),
                        'cl_loss', cl_loss.item(),
                        'debias_th:', round(self._current_debias_threshold(epoch), 4)
                    )

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            if epoch >= 5:
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type == 0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings,
            [self.data.user_num, self.data.item_num]
        )
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)

        view1 = torch.cat((user_view_1[u_idx], item_view_1[i_idx]), 0)
        view2 = torch.cat((user_view_2[u_idx], item_view_2[i_idx]), 0)

        return InfoNCE(view1, view2, self.temp)
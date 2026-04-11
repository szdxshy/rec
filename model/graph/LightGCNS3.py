import random
import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20


class LightGCNS3(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCNS3, self).__init__(conf, training_set, test_set)
        args = self.config['LightGCNS3']
        self.n_layers = int(args['n_layer'])

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

        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

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

            # keep current negative if it is acceptable
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

                batch_loss = (
                    bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                    + l2_reg_loss(
                        self.reg,
                        model.embedding_dict['user_emb'][user_idx],
                        model.embedding_dict['item_emb'][pos_idx],
                        model.embedding_dict['item_emb'][neg_idx]
                    ) / self.batch_size
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'batch_loss:', batch_loss.item(),
                        'debias_th:', round(self._current_debias_threshold(epoch), 4)
                    )

            with torch.no_grad():
                self.user_emb, self.item_emb = model()

            if epoch % 5 == 0:
                self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]

        return user_all_embeddings, item_all_embeddings
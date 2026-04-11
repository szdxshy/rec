import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE


class XSimGCLS7(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLS7, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCLS7']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])
        self.debias_temp_start = float(config.get('debias_temp_start', 0.95))
        self.debias_temp_end = float(config.get('debias_temp_end', 0.6))
        self.debias_pool = int(config.get('debias_pool', 20))

        # NEW: co-occurrence threshold
        self.cooc_threshold = int(config.get('cooc_threshold', 3))

        # NEW: precompute item-user sets for co-occurrence
        self.item_user_set = {}
        for u in self.data.training_set_u:
            for i in self.data.training_set_u[u]:
                self.item_user_set.setdefault(i, set()).add(u)

        # NEW: item popularity (for batch filtering)
        self.item_popularity = {i: len(self.item_user_set.get(i, [])) for i in self.data.item}

        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)

    def _current_debias_threshold(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.debias_temp_start + (self.debias_temp_end - self.debias_temp_start) * progress

    def _cooc_filter(self, pos_item, neg_item):
        pos_users = self.item_user_set.get(self.data.id2item[pos_item], set())
        neg_users = self.item_user_set.get(self.data.id2item[neg_item], set())
        return len(pos_users & neg_users) <= self.cooc_threshold

    def _remove_popular_batch(self, neg_list):
        # remove top-3 popular items in this batch
        pop_scores = [(i, self.item_popularity.get(self.data.id2item[i], 0)) for i in neg_list]
        pop_scores.sort(key=lambda x: x[1], reverse=True)
        remove_set = set([x[0] for x in pop_scores[:3]])
        return [i for i in neg_list if i not in remove_set]

    def _debias_negatives(self, user_idx, pos_idx, neg_idx, rec_user_emb, rec_item_emb, epoch):
        threshold = self._current_debias_threshold(epoch)
        item_keys = list(self.data.item.keys())
        debiased_neg = []

        for u, pos, neg in zip(user_idx, pos_idx, neg_idx):
            prob = torch.sigmoid(torch.matmul(rec_user_emb[u], rec_item_emb[neg])).item()

            # Co-occurrence filter
            if not self._cooc_filter(pos, neg):
                prob = 1.0  # force resample

            if prob <= threshold:
                debiased_neg.append(neg)
                continue

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

        # NEW: remove top-3 popular negatives in this batch
        debiased_neg = self._remove_popular_batch(debiased_neg)

        # If removing causes length mismatch, pad randomly
        while len(debiased_neg) < len(user_idx):
            debiased_neg.append(random.choice(debiased_neg))

        return debiased_neg

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)

                neg_idx = self._debias_negatives(user_idx, pos_idx, neg_idx,
                                                 rec_user_emb.detach(), rec_item_emb.detach(), epoch)

                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n,
                          'rec_loss:', rec_loss.item(),
                          'cl_loss', cl_loss.item(),
                          'debias_th:', round(self._current_debias_threshold(epoch), 4))

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.tensor(idx[0]).long()).cuda()
        i_idx = torch.unique(torch.tensor(idx[1]).long()).cuda()
        return InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp) + \
               InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1)).cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        return nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size)))
        })

    def forward(self, perturbed=False):
        ego = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings, cl_embeddings = [], ego

        for k in range(self.n_layers):
            ego = torch.sparse.mm(self.sparse_norm_adj, ego)

            if perturbed:
                noise = torch.rand_like(ego).cuda()
                ego += torch.sign(ego) * F.normalize(noise, dim=-1) * self.eps

            all_embeddings.append(ego)

            if k == self.layer_cl - 1:
                cl_embeddings = ego

        final = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        user_all, item_all = torch.split(final, [self.data.user_num, self.data.item_num])
        user_cl, item_cl = torch.split(cl_embeddings, [self.data.user_num, self.data.item_num])

        return (user_all, item_all, user_cl, item_cl) if perturbed else (user_all, item_all)
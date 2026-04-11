
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE


# XSimGCL S6: combine debiased negatives + co-occurrence positives
class XSimGCLS6(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLS6, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCLS6']

        # Original XSimGCL params
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])

        # Negative debias params
        self.debias_temp_start = float(config.get('debias_temp_start', 0.95))
        self.debias_temp_end = float(config.get('debias_temp_end', 0.6))
        self.debias_pool = int(config.get('debias_pool', 20))
        self.cooc_threshold = int(config.get('cooc_threshold', 3))

        # Positive sampling params
        self.soft_ratio_start = float(config.get('soft_ratio_start', 0.05))
        self.soft_ratio_end = float(config.get('soft_ratio_end', 0.2))
        self.pos_topk = int(config.get('pos_topk', 5))

        # Build item-user set for negative co-occurrence filtering
        self.item_user_set = {}
        for u in self.data.training_set_u:
            for i in self.data.training_set_u[u]:
                self.item_user_set.setdefault(i, set()).add(u)

        # Item popularity
        self.item_popularity = {
            i: len(self.item_user_set.get(i, []))
            for i in self.data.item
        }

        # Positive co-occurrence matrix
        self.cooc_norm = self._build_normalized_cooc()

        self.model = XSimGCL_Encoder(
            self.data,
            self.emb_size,
            self.eps,
            self.n_layers,
            self.layer_cl
        )

    def _build_normalized_cooc(self):
        interaction = self.data.interaction_mat.tocsr().astype(np.float32)
        item_user = interaction.transpose().tocsr()

        cooc = item_user.dot(item_user.transpose()).tocsr()

        freq = np.array(item_user.sum(axis=1)).flatten()
        freq = np.clip(freq, 1.0, None)
        inv_freq = 1.0 / freq

        d_inv = sp.diags(inv_freq)
        cooc_norm = d_inv.dot(cooc).dot(d_inv).tocsr()

        cooc_norm.setdiag(0.0)
        cooc_norm.eliminate_zeros()

        return cooc_norm

    def _current_debias_threshold(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.debias_temp_start + \
               (self.debias_temp_end - self.debias_temp_start) * progress

    def _current_soft_ratio(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.soft_ratio_start + \
               (self.soft_ratio_end - self.soft_ratio_start) * progress

    def _cooc_filter(self, pos_item, neg_item):
        pos_users = self.item_user_set.get(self.data.id2item[pos_item], set())
        neg_users = self.item_user_set.get(self.data.id2item[neg_item], set())
        return len(pos_users & neg_users) <= self.cooc_threshold

    def _sample_pos_by_cooc(self, pos_idx, epoch):
        ratio = self._current_soft_ratio(epoch)
        sampled_pos = []

        for pos in pos_idx:
            if random.random() > ratio:
                sampled_pos.append(pos)
                continue

            row = self.cooc_norm.getrow(pos)

            if row.nnz == 0:
                sampled_pos.append(pos)
                continue

            topk = min(self.pos_topk, row.nnz)
            top_indices = np.argsort(row.data)[-topk:]

            candidate_items = row.indices[top_indices]
            candidate_scores = row.data[top_indices]

            prob = candidate_scores / (candidate_scores.sum() + 1e-8)
            sampled = np.random.choice(candidate_items, p=prob)
            sampled_pos.append(int(sampled))

        return sampled_pos

    def _debias_negatives(self, user_idx, pos_idx, neg_idx,
                           rec_user_emb, rec_item_emb, epoch):
        threshold = self._current_debias_threshold(epoch)
        item_keys = list(self.data.item.keys())
        debiased_neg = []

        for u, pos, neg in zip(user_idx, pos_idx, neg_idx):
            prob = torch.sigmoid(
                torch.matmul(rec_user_emb[u], rec_item_emb[neg])
            ).item()

            # If co-occurrence too high, force resample
            if not self._cooc_filter(pos, neg):
                prob = 1.0

            if prob <= threshold:
                debiased_neg.append(neg)
                continue

            user_name = self.data.id2user[u]
            pool = []

            while len(pool) < self.debias_pool:
                cand = random.choice(item_keys)
                if cand not in self.data.training_set_u[user_name]:
                    pool.append(self.data.item[cand])

            pool_tensor = torch.tensor(pool).long().cuda()
            pool_scores = torch.sigmoid(
                torch.matmul(rec_item_emb[pool_tensor], rec_user_emb[u])
            )

            valid_candidates = []
            valid_scores = []

            for idx in range(len(pool)):
                candidate = pool[idx]
                score = pool_scores[idx].item()

                if score <= threshold and self._cooc_filter(pos, candidate):
                    valid_candidates.append(candidate)
                    valid_scores.append(score)

            if len(valid_candidates) > 0:
                sampled_neg = random.choice(valid_candidates)
                debiased_neg.append(sampled_neg)
            else:
                best_idx = torch.argmin(pool_scores).item()
                debiased_neg.append(pool[best_idx])

        return debiased_neg

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)

                sampled_pos = self._sample_pos_by_cooc(pos_idx, epoch)

                debiased_neg = self._debias_negatives(
                    user_idx,
                    sampled_pos,
                    neg_idx,
                    rec_user_emb.detach(),
                    rec_item_emb.detach(),
                    epoch
                )

                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[sampled_pos]
                neg_item_emb = rec_item_emb[debiased_neg]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                cl_loss = self.cl_rate * self.cal_cl_loss(
                    [user_idx, sampled_pos],
                    rec_user_emb,
                    cl_user_emb,
                    rec_item_emb,
                    cl_item_emb
                )

                batch_loss = rec_loss + \
                             l2_reg_loss(self.reg, user_emb, pos_item_emb) + \
                             cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'rec_loss:', rec_loss.item(),
                        'cl_loss:', cl_loss.item(),
                        'soft_ratio:', round(self._current_soft_ratio(epoch), 4),
                        'debias_th:', round(self._current_debias_threshold(epoch), 4)
                    )

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
        return torch.matmul(
            self.user_emb[u],
            self.item_emb.transpose(0, 1)
        ).cpu().numpy()


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
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(
            self.norm_adj
        ).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_

        return nn.ParameterDict({
            'user_emb': nn.Parameter(
                initializer(torch.empty(self.data.user_num, self.emb_size))
            ),
            'item_emb': nn.Parameter(
                initializer(torch.empty(self.data.item_num, self.emb_size))
            )
        })

    def forward(self, perturbed=False):
        ego = torch.cat([
            self.embedding_dict['user_emb'],
            self.embedding_dict['item_emb']
        ], 0)

        all_embeddings = []
        cl_embeddings = ego

        for k in range(self.n_layers):
            ego = torch.sparse.mm(self.sparse_norm_adj, ego)

            if perturbed:
                noise = torch.rand_like(ego).cuda()
                ego += torch.sign(ego) * F.normalize(noise, dim=-1) * self.eps

            all_embeddings.append(ego)

            if k == self.layer_cl - 1:
                cl_embeddings = ego

        final = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        user_all, item_all = torch.split(
            final,
            [self.data.user_num, self.data.item_num]
        )

        user_cl, item_cl = torch.split(
            cl_embeddings,
            [self.data.user_num, self.data.item_num]
        )

        return (user_all, item_all, user_cl, item_cl) if perturbed else (user_all, item_all)



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


class XSimGCLS4(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLS4, self).__init__(conf, training_set, test_set)

        config = self.config['XSimGCLS4']

        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])

        # negative sampling
        self.debias_pool = int(config.get('debias_pool', 50))
        self.cooc_threshold = int(config.get('cooc_threshold', 3))
        self.neg_top_ratio = float(config.get('neg_top_ratio', 0.2))
        self.neg_bottom_ratio = float(config.get('neg_bottom_ratio', 0.3))
        self.max_repeat_per_item = int(config.get('max_repeat_per_item', 2))

        # positive sampling
        self.soft_ratio_start = float(config.get('soft_ratio_start', 0.02))
        self.soft_ratio_end = float(config.get('soft_ratio_end', 0.10))
        self.cooc_topk = int(config.get('cooc_topk', 10))
        self.cooc_sim_threshold = float(config.get('cooc_sim_threshold', 0.02))
        self.soft_mix_alpha = float(config.get('soft_mix_alpha', 0.7))

        # item-user sets
        self.item_user_set = {}
        for u in self.data.training_set_u:
            for i in self.data.training_set_u[u]:
                self.item_user_set.setdefault(i, set()).add(u)

        self.item_keys = list(self.data.item.keys())

        self.model = XSimGCL_Encoder(
            self.data,
            self.emb_size,
            self.eps,
            self.n_layers,
            self.layer_cl
        )

        self.cooc_norm = self._build_normalized_cooc()

    def _build_normalized_cooc(self):
        interaction = self.data.interaction_mat.tocsr().astype(np.float32)
        item_user = interaction.transpose().tocsr()

        cooc = item_user.dot(item_user.transpose()).tocsr()

        freq = np.array(item_user.sum(axis=1)).flatten()
        freq = np.clip(freq, 1.0, None)

        inv_sqrt_freq = 1.0 / np.sqrt(freq)
        d_inv = sp.diags(inv_sqrt_freq)

        cooc_norm = d_inv.dot(cooc).dot(d_inv).tocsr()
        cooc_norm.setdiag(0.0)
        cooc_norm.eliminate_zeros()

        return cooc_norm

    def _cooc_filter(self, pos_item, neg_item):
        pos_users = self.item_user_set.get(self.data.id2item[pos_item], set())
        neg_users = self.item_user_set.get(self.data.id2item[neg_item], set())
        return len(pos_users & neg_users) <= self.cooc_threshold

    def _current_soft_ratio(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.soft_ratio_start + (self.soft_ratio_end - self.soft_ratio_start) * progress

    def _sample_candidate_pool(self, user_name, pool_size):
        pool = []
        user_hist = self.data.training_set_u[user_name]

        while len(pool) < pool_size:
            cand = random.choice(self.item_keys)
            if cand not in user_hist:
                pool.append(self.data.item[cand])

        return pool

    def _select_percentile_negatives(self, scores, candidates):
        scores_np = scores.detach().cpu().numpy()
        sorted_idx = np.argsort(scores_np)
        n = len(sorted_idx)

        low_cut = int(n * self.neg_bottom_ratio)
        high_cut = int(n * (1.0 - self.neg_top_ratio))

        selected_idx = sorted_idx[low_cut:high_cut]

        if len(selected_idx) == 0:
            selected_idx = sorted_idx[:max(1, n // 2)]

        return [candidates[idx] for idx in selected_idx]

    def _debias_negatives(self, user_idx, pos_idx, rec_user_emb, rec_item_emb):
        debiased_neg = []
        neg_counter = {}

        for u, pos in zip(user_idx, pos_idx):
            user_name = self.data.id2user[u]
            pool = self._sample_candidate_pool(user_name, self.debias_pool)

            filtered_pool = []
            for cand in pool:
                if self._cooc_filter(pos, cand):
                    filtered_pool.append(cand)

            if len(filtered_pool) == 0:
                filtered_pool = pool

            pool_scores = torch.sigmoid(
                torch.matmul(rec_item_emb[filtered_pool], rec_user_emb[u])
            )

            valid_candidates = self._select_percentile_negatives(
                pool_scores,
                filtered_pool
            )

            diverse_candidates = []
            for cand in valid_candidates:
                if neg_counter.get(cand, 0) < self.max_repeat_per_item:
                    diverse_candidates.append(cand)

            if len(diverse_candidates) > 0:
                chosen = random.choice(diverse_candidates)
            elif len(valid_candidates) > 0:
                chosen = random.choice(valid_candidates)
            else:
                chosen = filtered_pool[torch.argmin(pool_scores).item()]

            debiased_neg.append(chosen)
            neg_counter[chosen] = neg_counter.get(chosen, 0) + 1

        return debiased_neg

    def _sample_soft_positive(self, pos_idx, epoch):
        ratio = self._current_soft_ratio(epoch)

        sampled_pos = []
        use_soft_mask = []

        for pos in pos_idx:
            if random.random() > ratio:
                sampled_pos.append(pos)
                use_soft_mask.append(False)
                continue

            row = self.cooc_norm.getrow(pos)

            if row.nnz == 0:
                sampled_pos.append(pos)
                use_soft_mask.append(False)
                continue

            valid_mask = row.data >= self.cooc_sim_threshold

            if valid_mask.sum() == 0:
                sampled_pos.append(pos)
                use_soft_mask.append(False)
                continue

            valid_scores = row.data[valid_mask]
            valid_items = row.indices[valid_mask]

            topk = min(self.cooc_topk, len(valid_items))
            top_idx = np.argpartition(valid_scores, -topk)[-topk:]

            candidate_items = valid_items[top_idx]
            candidate_scores = valid_scores[top_idx]

            probs = candidate_scores / candidate_scores.sum()
            sampled = np.random.choice(candidate_items, p=probs)

            sampled_pos.append(int(sampled))
            use_soft_mask.append(True)

        return sampled_pos, use_soft_mask

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch

                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)

                neg_idx = self._debias_negatives(
                    user_idx,
                    pos_idx,
                    rec_user_emb.detach(),
                    rec_item_emb.detach()
                )

                sampled_pos, use_soft_mask = self._sample_soft_positive(
                    pos_idx,
                    epoch
                )

                user_emb = rec_user_emb[user_idx]
                orig_pos_emb = rec_item_emb[pos_idx]
                soft_pos_emb = rec_item_emb[sampled_pos]
                neg_item_emb = rec_item_emb[neg_idx]

                pos_item_emb = orig_pos_emb.clone()

                use_soft_mask_tensor = torch.tensor(use_soft_mask).bool().cuda()
                alpha = self.soft_mix_alpha

                pos_item_emb[use_soft_mask_tensor] = (
                    alpha * orig_pos_emb[use_soft_mask_tensor] +
                    (1.0 - alpha) * soft_pos_emb[use_soft_mask_tensor]
                )

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                cl_loss = self.cl_rate * self.cal_cl_loss(
                    [user_idx, sampled_pos],
                    rec_user_emb,
                    cl_user_emb,
                    rec_item_emb,
                    cl_item_emb
                )

                batch_loss = (
                    rec_loss +
                    l2_reg_loss(self.reg, user_emb, pos_item_emb) +
                    cl_loss
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'rec_loss:', rec_loss.item(),
                        'cl_loss:', cl_loss.item(),
                        'soft_ratio:', round(self._current_soft_ratio(epoch), 4)
                    )

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.tensor(idx[0]).long()).cuda()
        i_idx = torch.unique(torch.tensor(idx[1]).long()).cuda()

        return (
            InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp) +
            InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        )

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
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size)))
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

        user_all, item_all = torch.split(final, [self.data.user_num, self.data.item_num])
        user_cl, item_cl = torch.split(cl_embeddings, [self.data.user_num, self.data.item_num])

        return (user_all, item_all, user_cl, item_cl) if perturbed else (user_all, item_all)


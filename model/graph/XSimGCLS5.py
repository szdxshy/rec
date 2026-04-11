
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


class XSimGCLS5(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLS5, self).__init__(conf, training_set, test_set)

        config = self.config['XSimGCLS5']

        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])

        # softer schedule
        self.soft_ratio_start = float(config.get('soft_ratio_start', 0.02))
        self.soft_ratio_end = float(config.get('soft_ratio_end', 0.10))

        # top-k co-occurrence candidates
        self.cooc_topk = int(config.get('cooc_topk', 10))

        # minimum similarity threshold
        self.cooc_sim_threshold = float(config.get('cooc_sim_threshold', 0.02))

        # embedding mix weight
        self.soft_mix_alpha = float(config.get('soft_mix_alpha', 0.7))

        self.model = XSimGCL_Encoder(
            self.data,
            self.emb_size,
            self.eps,
            self.n_layers,
            self.layer_cl
        )

        self.cooc_norm = self._build_normalized_cooc()

    def _build_normalized_cooc(self):
        interaction = self.data.interaction_mat.tocsr().astype(np.float32)  # U x I
        item_user = interaction.transpose().tocsr()  # I x U

        cooc = item_user.dot(item_user.transpose()).tocsr()  # I x I

        freq = np.array(item_user.sum(axis=1)).flatten()
        freq = np.clip(freq, 1.0, None)

        # cosine normalization: D^{-1/2} A D^{-1/2}
        inv_sqrt_freq = 1.0 / np.sqrt(freq)
        d_inv = sp.diags(inv_sqrt_freq)

        cooc_norm = d_inv.dot(cooc).dot(d_inv).tocsr()

        cooc_norm.setdiag(0.0)
        cooc_norm.eliminate_zeros()

        return cooc_norm

    def _current_soft_ratio(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.soft_ratio_start + (self.soft_ratio_end - self.soft_ratio_start) * progress

    def _sample_soft_positive(self, pos_idx, epoch):
        ratio = self._current_soft_ratio(epoch)

        sampled_pos = []
        use_soft_mask = []

        for pos in pos_idx:
            # default: keep original positive
            if random.random() > ratio:
                sampled_pos.append(pos)
                use_soft_mask.append(False)
                continue

            row = self.cooc_norm.getrow(pos)

            if row.nnz == 0:
                sampled_pos.append(pos)
                use_soft_mask.append(False)
                continue

            # similarity filter
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
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)

                sampled_pos, use_soft_mask = self._sample_soft_positive(pos_idx, epoch)

                user_emb = rec_user_emb[user_idx]
                orig_pos_emb = rec_item_emb[pos_idx]
                soft_pos_emb = rec_item_emb[sampled_pos]
                neg_item_emb = rec_item_emb[neg_idx]

                # mixed positive embedding
                alpha = self.soft_mix_alpha
                pos_item_emb = orig_pos_emb.clone()

                use_soft_mask_tensor = torch.tensor(use_soft_mask).bool().cuda()

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

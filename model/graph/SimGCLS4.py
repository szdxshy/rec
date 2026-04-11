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

# SimGCLS4: random negatives + co-occurrence based positives
# only positive sampling strategy is changed

class SimGCLS4(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCLS4, self).__init__(conf, training_set, test_set)
        args = self.config['SimGCLS4']

        self.cl_rate = float(args['lambda'])
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])

        self.soft_ratio_start = float(args.get('soft_ratio_start', 0.05))
        self.soft_ratio_end = float(args.get('soft_ratio_end', 0.2))

        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        self.cooc_norm = self._build_normalized_cooc()

    def _build_normalized_cooc(self):
        interaction = self.data.interaction_mat.tocsr().astype(np.float32)  # U x I
        item_user = interaction.transpose().tocsr()  # I x U

        # item-item co-occurrence
        cooc = item_user.dot(item_user.transpose()).tocsr()

        # normalize by item frequency
        freq = np.array(item_user.sum(axis=1)).flatten()
        freq = np.clip(freq, 1.0, None)
        inv_freq = 1.0 / freq
        d_inv = sp.diags(inv_freq)

        cooc_norm = d_inv.dot(cooc).dot(d_inv).tocsr()
        cooc_norm.setdiag(0.0)
        cooc_norm.eliminate_zeros()

        return cooc_norm

    def _current_soft_ratio(self, epoch):
        span = max(1, self.maxEpoch - 1)
        progress = epoch / span
        return self.soft_ratio_start + (self.soft_ratio_end - self.soft_ratio_start) * progress

    def _sample_pos_by_cooc(self, pos_idx, epoch):
        ratio = self._current_soft_ratio(epoch)
        sampled_pos = []

        for pos in pos_idx:
            # keep original positive with probability (1 - ratio)
            if random.random() > ratio:
                sampled_pos.append(pos)
                continue

            row = self.cooc_norm.getrow(pos)

            # no co-occurrence neighbor
            if row.nnz == 0:
                sampled_pos.append(pos)
                continue

            # replace with strongest co-occurrence item
            best_local = int(np.argmax(row.data))
            sampled_pos.append(int(row.indices[best_local]))

        return sampled_pos

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                sampled_pos = self._sample_pos_by_cooc(pos_idx, epoch)

                rec_user_emb, rec_item_emb = model()

                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[sampled_pos]
                neg_item_emb = rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, sampled_pos])

                batch_loss = (
                    rec_loss
                    + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                    + cl_loss
                )

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'rec_loss:', rec_loss.item(),
                        'cl_loss', cl_loss.item(),
                        'soft_ratio:', round(self._current_soft_ratio(epoch), 4)
                    )

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            self.fast_evaluation(epoch)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)

        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
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

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat(
            [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0
        )

        all_embeddings = []

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += (
                    torch.sign(ego_embeddings)
                    * F.normalize(random_noise, dim=-1)
                    * self.eps
                )

            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings,
            [self.data.user_num, self.data.item_num]
        )

        return user_all_embeddings, item_all_embeddings
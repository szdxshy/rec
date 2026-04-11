import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss

# LightGCNS4: random negatives + co-occurrence based positives
# only positive sampling strategy is changed

class LightGCNS4(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCNS4, self).__init__(conf, training_set, test_set)
        args = self.config['LightGCNS4']
        self.n_layers = int(args['n_layer'])

        self.soft_ratio_start = float(args.get('soft_ratio_start', 0.05))
        self.soft_ratio_end = float(args.get('soft_ratio_end', 0.2))

        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
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

                batch_loss = (
                    bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                    + l2_reg_loss(
                        self.reg,
                        model.embedding_dict['user_emb'][user_idx],
                        model.embedding_dict['item_emb'][sampled_pos],
                        model.embedding_dict['item_emb'][neg_idx]
                    ) / self.batch_size
                )

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(
                        'training:', epoch + 1,
                        'batch', n,
                        'batch_loss:', batch_loss.item(),
                        'soft_ratio:', round(self._current_soft_ratio(epoch), 4)
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
        ego_embeddings = torch.cat(
            [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0
        )

        all_embeddings = [ego_embeddings]

        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]

        return user_all_embeddings, item_all_embeddings
# LightGCNS2

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss

# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20
# LightGCNS2:
# only modifies positive / negative sampling strategy
# keeps original LightGCN encoder unchanged


class LightGCNS2(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCNS2, self).__init__(conf, training_set, test_set)
        args = self.config['LightGCNS2']
        self.n_layers = int(args['n_layer'])

        # ===== sampling hyper-parameters =====
        self.pos_topk = int(args['pos_topk']) if 'pos_topk' in args else 10
        self.soft_ratio = float(args['soft_ratio']) if 'soft_ratio' in args else 0.3
        self.debias_threshold = float(args['debias_threshold']) if 'debias_threshold' in args else 0.1

        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

        self.user_pos_items = defaultdict(set)
        self.item_cooc = self.build_item_cooccurrence()

    def build_item_cooccurrence(self):
        item_cooc = defaultdict(dict)

        for u in self.data.training_set_u:
            items = list(self.data.training_set_u[u].keys())
            for i in items:
                for j in items:
                    if i == j:
                        continue
                    item_cooc[i][j] = item_cooc[i].get(j, 0) + 1

        for u in self.data.training_set_u:
            uid = self.data.user[u]
            self.user_pos_items[uid] = set([
                self.data.item[i] for i in self.data.training_set_u[u].keys()
            ])

        return item_cooc

    def sample_positive_item(self, pos_item):
        if pos_item not in self.item_cooc or len(self.item_cooc[pos_item]) == 0:
            return pos_item

        neighbors = sorted(
            self.item_cooc[pos_item].items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.pos_topk]

        items = [x[0] for x in neighbors]
        weights = np.array([x[1] for x in neighbors], dtype=np.float32)
        weights = weights / weights.sum()

        sampled_item = np.random.choice(items, p=weights)
        return sampled_item

    def sample_negative_item(self, user, pos_item, neg_item, cur_threshold):
        user_hist = self.user_pos_items[user]

        if pos_item in self.item_cooc:
            cooc_items = self.item_cooc[pos_item]
        else:
            cooc_items = {}

        candidate_neg = neg_item
        retry = 0

        while retry < 20:
            if candidate_neg not in user_hist:
                cooc_score = cooc_items.get(candidate_neg, 0)
                if cooc_score <= cur_threshold:
                    break

            candidate_neg = np.random.randint(0, self.data.item_num)
            retry += 1

        return candidate_neg

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            cur_soft_ratio = min(0.8, self.soft_ratio + epoch * 0.02)
            cur_debias_threshold = self.debias_threshold + epoch * 0.01

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                enhanced_pos_idx = []
                enhanced_neg_idx = []

                for u, p, neg in zip(user_idx, pos_idx, neg_idx):
                    if np.random.rand() < cur_soft_ratio:
                        new_pos = self.sample_positive_item(p)
                    else:
                        new_pos = p

                    new_neg = self.sample_negative_item(
                        u,
                        p,
                        neg,
                        cur_debias_threshold
                    )

                    enhanced_pos_idx.append(new_pos)
                    enhanced_neg_idx.append(new_neg)

                pos_idx = np.array(enhanced_pos_idx)
                neg_idx = np.array(enhanced_neg_idx)

                rec_user_emb, rec_item_emb = model()
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
                        'soft_ratio', round(cur_soft_ratio, 4),
                        'debias_threshold', round(cur_debias_threshold, 4)
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
        ego_embeddings = torch.cat([
            self.embedding_dict['user_emb'],
            self.embedding_dict['item_emb']
        ], 0)

        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


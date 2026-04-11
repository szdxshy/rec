import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE


# XSimGCL S3: based on S2 but without confidence-weighted loss
class XSimGCLS3(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLS3, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCLS3']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])
        self.soft_ratio = float(config.get('soft_ratio', 0.2))
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)

    def _sample_pos(self, user_idx, pos_idx, rec_item_emb, epoch):
        if epoch < self.maxEpoch // 2:
            return pos_idx
        sampled_pos = []
        item_emb = F.normalize(rec_item_emb.detach(), dim=1)
        for _, pos in zip(user_idx, pos_idx):
            if random.random() > self.soft_ratio:
                sampled_pos.append(pos)
                continue
            item_name = self.data.id2item[pos]
            interacted_users = list(self.data.training_set_i[item_name].keys())
            if len(interacted_users) == 0:
                sampled_pos.append(pos)
                continue
            anchor_users = sorted(interacted_users, key=lambda x: len(self.data.training_set_u[x]))[:2]
            candidate_items = set()
            for user_name in anchor_users:
                candidate_items.update(self.data.training_set_u[user_name].keys())
            candidate_ids = [self.data.item[i] for i in candidate_items if self.data.item[i] != pos]
            if len(candidate_ids) == 0:
                sampled_pos.append(pos)
                continue
            sim = torch.matmul(item_emb[candidate_ids], item_emb[pos])
            sampled_pos.append(candidate_ids[torch.argmax(sim).item()])
        return sampled_pos

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                sampled_pos = self._sample_pos(user_idx, pos_idx, rec_item_emb, epoch)
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[sampled_pos]
                neg_item_emb = rec_item_emb[neg_idx]

                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, sampled_pos], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad(); batch_loss.backward(); optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.tensor(idx[0]).long()).cuda()
        i_idx = torch.unique(torch.tensor(idx[1]).long()).cuda()
        return InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp) + InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1)).cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data; self.eps = eps; self.emb_size = emb_size; self.n_layers = n_layers; self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        return nn.ParameterDict({'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
                                 'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size)))})

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

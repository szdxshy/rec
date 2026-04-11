import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import weighted_bpr_loss, l2_reg_loss, InfoNCE


# XSimGCL + staged S2 positive sampling
class XSimGCLS2(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLS2, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCLS2']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])
        self.soft_ratio_start = float(config.get('soft_ratio_start', 0.6))
        self.soft_ratio_end = float(config.get('soft_ratio_end', 0.4))
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)


    def _current_soft_ratio(self, epoch):
        half = self.maxEpoch // 2
        if epoch < half:
            return 0.0
        span = max(1, self.maxEpoch - half - 1)
        progress = (epoch - half) / span
        return self.soft_ratio_start + (self.soft_ratio_end - self.soft_ratio_start) * progress

    def _sample_pos_with_conf(self, user_idx, pos_idx, rec_item_emb, epoch):
        ratio = self._current_soft_ratio(epoch)
        if ratio <= 0:
            conf = torch.ones(len(pos_idx), device=rec_item_emb.device)
            return pos_idx, conf

        sampled_pos, conf_list = [], []
        item_emb = F.normalize(rec_item_emb.detach(), dim=1)
        for u, pos in zip(user_idx, pos_idx):
            if random.random() > ratio:
                sampled_pos.append(pos)
                conf_list.append(1.0)
                continue

            item_name = self.data.id2item[pos]
            interacted_users = list(self.data.training_set_i[item_name].keys())
            if len(interacted_users) == 0:
                sampled_pos.append(pos)
                conf_list.append(1.0)
                continue

            anchor_users = sorted(interacted_users, key=lambda x: len(self.data.training_set_u[x]))[:2]
            candidate_items = set()
            for user_name in anchor_users:
                candidate_items.update(self.data.training_set_u[user_name].keys())

            candidate_ids = [self.data.item[i] for i in candidate_items if self.data.item[i] != pos]
            if len(candidate_ids) == 0:
                sampled_pos.append(pos)
                conf_list.append(1.0)
                continue

            sim = torch.matmul(item_emb[candidate_ids], item_emb[pos])
            best_idx = torch.argmax(sim).item()
            soft_pos = candidate_ids[best_idx]
            confidence = ((sim[best_idx].item() + 1.0) / 2.0)

            sampled_pos.append(soft_pos)
            conf_list.append(confidence)

        conf = torch.tensor(conf_list, dtype=rec_item_emb.dtype, device=rec_item_emb.device)
        return sampled_pos, conf

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)

                sampled_pos, conf = self._sample_pos_with_conf(user_idx, pos_idx, rec_item_emb, epoch)
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[sampled_pos]
                neg_item_emb = rec_item_emb[neg_idx]

                rec_loss = weighted_bpr_loss(user_emb, pos_item_emb, neg_item_emb, conf)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, sampled_pos], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


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
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

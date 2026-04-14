import random
import torch
from model.graph.XSimGCL import XSimGCL, XSimGCL_Encoder
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss


class XSimGCL_dyn(XSimGCL):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCL_dyn']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)

    def _sample_random_negative(self, user):
        neg = random.randint(0, self.data.item_num - 1)
        while self.data.id2item[neg] in self.data.training_set_u[user]:
            neg = random.randint(0, self.data.item_num - 1)
        return neg

    def _sample_dynamic_negatives(self, user_idx, rec_user_emb, rec_item_emb, epoch, pool_size=128):
        hard_ratio = min(1.0, float(epoch + 1) / float(self.maxEpoch))
        neg_idx = []
        for u in user_idx:
            user = self.data.id2user[u]
            if random.random() < hard_ratio:
                candidates = []
                while len(candidates) < pool_size:
                    candidates.append(self._sample_random_negative(user))
                candidates = torch.tensor(candidates, device=rec_item_emb.device, dtype=torch.long)
                scores = torch.matmul(rec_item_emb[candidates], rec_user_emb[u])
                neg_idx.append(candidates[torch.argmax(scores)].item())
            else:
                neg_idx.append(self._sample_random_negative(user))
        return neg_idx

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                neg_idx = self._sample_dynamic_negatives(user_idx, rec_user_emb, rec_item_emb, epoch)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
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

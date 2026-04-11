import random
import torch
from model.graph.SGL import SGL, SGL_Encoder
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss


class SGL_neg(SGL):
    def __init__(self, conf, training_set, test_set):
        super(SGL, self).__init__(conf, training_set, test_set)
        args = self.config['SGL_neg']
        self.cl_rate = float(args['lambda'])
        aug_type = self.aug_type = int(args['aug_type'])
        drop_rate = float(args['drop_rate'])
        n_layers = int(args['n_layer'])
        temp = float(args['temp'])
        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp, aug_type)

    def _sample_hard_negatives(self, user_idx, rec_user_emb, rec_item_emb, pool_size=128):
        neg_idx = []
        for u in user_idx:
            user = self.data.id2user[u]
            user_hist = self.data.training_set_u[user]
            candidates = []
            while len(candidates) < pool_size:
                cand = random.randint(0, self.data.item_num - 1)
                if self.data.id2item[cand] not in user_hist:
                    candidates.append(cand)
            candidates = torch.tensor(candidates, device=rec_item_emb.device, dtype=torch.long)
            scores = torch.matmul(rec_item_emb[candidates], rec_user_emb[u])
            neg_idx.append(candidates[torch.argmax(scores)].item())
        return neg_idx

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            dropped_adj1 = model.graph_reconstruction()
            dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch
                rec_user_emb, rec_item_emb = model()
                neg_idx = self._sample_hard_negatives(user_idx, rec_user_emb, rec_item_emb)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], dropped_adj1, dropped_adj2)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch >= 5:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

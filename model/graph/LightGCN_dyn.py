import random
import torch
from model.graph.LightGCN import LightGCN, LGCN_Encoder
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss


class LightGCN_dyn(LightGCN):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = self.config['LightGCN_dyn']
        self.n_layers = int(args['n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

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
                    cand = self._sample_random_negative(user)
                    candidates.append(cand)
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
                rec_user_emb, rec_item_emb = model()
                neg_idx = self._sample_dynamic_negatives(user_idx, rec_user_emb, rec_item_emb, epoch)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx], model.embedding_dict['item_emb'][pos_idx], model.embedding_dict['item_emb'][neg_idx]) / self.batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

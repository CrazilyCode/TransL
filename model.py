import torch
import torch.nn as nn

torch.manual_seed(1)

class Network(nn.Module):

    def __init__(self, entity_size, dimension, rel_size):
        super(Network, self).__init__()
        self.eneity_embedding = nn.Embedding(entity_size + 1, dimension)
        self.edge_weight = nn.Embedding(rel_size * (rel_size + 1) + 1, 1)
        self.rel_embedding = nn.Embedding(rel_size, dimension)
        
        self.softmax = nn.Softmax(dim=1)

    def get_t(self, data_r, data_e, rel):
        data_r = self.edge_weight(data_r.long())
        data_r = data_r.view(data_e.size(0), -1)
        data_w = self.softmax(data_r)
        data_w = data_w.view(data_e.size(0), -1, 1)

        data_e = self.eneity_embedding(data_e.long())

        Eh = torch.mul(data_e, data_w)
        Eh = torch.transpose(Eh, 1, 2)
        Eh = torch.sum(Eh, dim=2)

        r = self.rel_embedding(rel.long())
        out_t = Eh + r

        return out_t

    def forward(self, data_r, data_e, rel, pos_id, neg_id):
        out_t = self.get_t(data_r, data_e, rel)

        pos_out = self.eneity_embedding(pos_id.long())
        neg_out = self.eneity_embedding(neg_id.long())

        return out_t, pos_out, neg_out


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin1, margin2):
        super(ContrastiveLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, out_self, pos_out, neg_out):
        pdist = nn.PairwiseDistance(p=2)
        pos_dist = pdist(out_self, pos_out)
        neg_dist = pdist(out_self, neg_out)
        loss = torch.mean(torch.clamp(pos_dist - self.margin1, min=0.0) + torch.clamp(self.margin2 - neg_dist, min=0.0))
        # loss = torch.mean(torch.clamp(pos_dist - neg_dist, min=0.0))

        return loss

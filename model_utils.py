import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class RelationAttention(nn.Module):
    def __init__(self, in_dim = 300, hidden_dim = 64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        return out  # ([N, L])


class LinearAttention(nn.Module):
    '''
    re-implement of gat's attention
    '''
    def __init__(self, in_dim = 300, mem_dim = 300):
        # in dim, the dimension of query vector
        super().__init__()
        self.linear = nn.Linear(in_dim, mem_dim)
        self.fc = nn.Linear(mem_dim * 2, 1)
        self.leakyrelu = nn.LeakyReLU(1e-2)

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = self.linear(aspect_v) # (N, D)
        Q = Q.unsqueeze(1)  # (N, 1, D)
        Q = Q.expand_as(feature) # (N, L, D)
        Q = self.linear(Q) # (N, L, D)
        feature = self.linear(feature) # (N, L, D)

        att_feature = torch.cat([feature, Q], dim = 2) # (N, L, 2D)
        att_weight = self.fc(att_feature) # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, L, 1)
        att_weight = mask_logits(att_weight, dmask)  # (N, L ,1)

        attention = F.softmax(att_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)

        return out


class DotprodAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, aspect_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, D]
        mask dmask          [N, L]
        '''

        Q = aspect_v
        Q = Q.unsqueeze(2)  # (N, D, 1)
        dot_prod = torch.bmm(feature, Q)  # (N, L, 1)
        dmask = dmask.unsqueeze(2)  # (N, D, 1)
        attention_weight = mask_logits(dot_prod, dmask)  # (N, L ,1)
        attention = F.softmax(attention_weight, dim=1)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), attention)  # (N, D, 1)
        out = out.squeeze(2)
        # out = F.sigmoid(out)
        # (N, D), ([N, L]), (N, L, 1)
        return out

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class DepparseMultiHeadAttention(nn.Module):
    def __init__(self, h=6, Co=300, cat=True):
        super().__init__()
        self.hidden_size = Co // h
        self.h = h
        self.fc1 = nn.Linear(Co, Co)
        self.relu = nn.ReLU()
        self.fc2s = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(h)])
        self.cat = cat

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        nbatches = dep_tags_v.size(0)
        Q = self.fc1(dep_tags_v).view(nbatches, -1, self.h,
                                      self.hidden_size)  # [N, L, #heads, hidden_size]
        Q = self.relu(Q)
        Q = Q.transpose(0, 2)  # [#heads, L, N, hidden_size]
        Q = [l(q).squeeze(2).transpose(0, 1)
             for l, q in zip(self.fc2s, Q)]  # [N, L] * #heads
        # Q = Q.squeeze(2)
        Q = [F.softmax(mask_logits(q, dmask), dim=1).unsqueeze(2)
             for q in Q]  # [N, L, 1] * #heads

        # Q = Q.unsqueeze(2)
        if self.cat:
            out = torch.cat(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=1)
        else:
            out = torch.stack(
                [torch.bmm(feature.transpose(1, 2), q).squeeze(2) for q in Q], dim=2)
            out = torch.sum(out, dim=2)
        # out = out.squeeze(2)
        return out, Q[0]  # ([N, L]) one head



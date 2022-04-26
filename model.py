import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphAttentionLayer
from neigborMatirx import generate_neighbor_list


class DGAT(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, dropout,  adj):
        """Dense version of GAT."""
        super(DGAT, self).__init__()
        self.dropout = dropout
        self.nhid = nhidden
        # print(nhead)
        self.heads = 4

        self.W = nn.Parameter(torch.empty(
            size=(self.heads, nfeat, nhidden)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attentions = self._build_muti_head_list(adj)
        self.fc = nn.Linear(nhidden * self.heads, nclass)

    def _build_muti_head_list(self, adj):
        neighbor_lists = generate_neighbor_list(adj, 3)
        layer_list = [
            GraphAttentionLayer(self.nhid, self.dropout, neighbor_lists[0], False),
            GraphAttentionLayer(self.nhid, self.dropout, neighbor_lists[1], False),
            GraphAttentionLayer(self.nhid, self.dropout, neighbor_lists[2], False)
            ]
        return nn.ModuleList(layer_list)

    def forward(self, x):
        x =torch.cat([torch.mm(x,self.W[self.heads-1])]+[
            self.attentions[i](
                torch.mm(x, self.W[i])
            ) for i in range(self.heads-1)
        ],dim=1)

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = (self.fc(x))
        return F.log_softmax(x, dim=1)

from sklearn import neighbors
import torch.nn as nn
import torch
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, nhid, dropout, neighbor_list, self_loop=True):
        """Dense version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.neighbor_list = neighbor_list
        self.self_loop = self_loop
        self.scare = nhid**0.5
        self.Q = nn.Parameter(torch.empty(
            size=(nhid, nhid)))
        nn.init.xavier_uniform_(self.Q.data, gain=1.414)
        self.K = nn.Parameter(torch.empty(
            size=(nhid, nhid)))
        nn.init.xavier_uniform_(self.K.data, gain=1.414)
        self.V = nn.Parameter(torch.empty(
            size=(nhid, nhid)))
        nn.init.xavier_uniform_(self.V.data, gain=1.414)

        # self.Q1 = nn.Parameter(torch.empty(
        #     size=(nhid, nhid)))
        # nn.init.xavier_uniform_(self.Q1.data, gain=1.414)
        # self.K1 = nn.Parameter(torch.empty(
        #     size=(nhid, nhid)))
        # nn.init.xavier_uniform_(self.K1.data, gain=1.414)
        # self.V1 = nn.Parameter(torch.empty(
        #     size=(nhid, nhid)))
        # nn.init.xavier_uniform_(self.V1.data, gain=1.414)
        self.mask1, self.mask2 = self._generate_attention_mask()

    def _generate_attention_mask(self):
        n, m = self.neighbor_list.shape
        mask = torch.where(self.neighbor_list == n-1, torch.zeros_like(
            self.neighbor_list), torch.ones_like(self.neighbor_list)).unsqueeze(1)
        mask1 = torch.bmm(mask, mask.transpose(-1, -2))
        if self.self_loop:
            mask2 = torch.cat([torch.ones(n, 1, 1), mask], dim=2)
        else:
            mask2 = mask
        return mask1, mask2

    def _neighbor_attention(self, neighbor_matrix):
        Q = torch.einsum('ijk,km->ijm', neighbor_matrix, self.Q)
        K = torch.einsum('ijk,km->ijm', neighbor_matrix, self.K)
        V = torch.einsum('ijk,km->ijm', neighbor_matrix, self.V)

        # Q=F.normalize(Q)
        # K=F.normalize(K)
        attention_score = torch.bmm(Q,
                                    K.transpose(-1, -2))
        attention_score = F.leaky_relu(attention_score)
        attention_score = torch.where(
            self.mask1 > 0, (attention_score), torch.ones_like(attention_score)*-9e15)*self.scare
        attention_score = F.softmax(attention_score, dim=2)
        # print(attention_score)
        attention_score = F.dropout(
            attention_score, self.dropout, training=self.training)
        neighbor_attention_res = torch.bmm(attention_score, V)

        return neighbor_attention_res  # shape=n,(neighbor_number),nhid

    def _attention_layer(self, x, neighbor_attention_res):
        x = x.reshape(-1, 1, self.nhid)
        if self.self_loop:
            neighbor_attention_res = torch.cat(
                [x, neighbor_attention_res], dim=1)
        Q = torch.einsum('ijk,km->ijm', x, self.Q)  # n,1,nhid
        K = torch.einsum('ijk,km->ijm', neighbor_attention_res,
                         self.K)  # n,(neighbor_number+1),nhid
        V = torch.einsum('ijk,km->ijm', neighbor_attention_res,
                         self.V)  # n,(neighbor_number+1),nhid
        attention_score = torch.bmm(Q,
                                    K.transpose(-1, -2))*self.scare  # n,1,(neighbor_number+1)
        attention_score = F.leaky_relu(attention_score)
        attention_score = torch.where(
            self.mask2 > 0, (attention_score), torch.ones_like(attention_score)*-9e15)
        attention_score = F.softmax(attention_score, dim=2)
        attention_score = F.dropout(
            attention_score, self.dropout, training=self.training)
        attention_res = torch.bmm(attention_score, V)
        return attention_res.squeeze(1)

    def forward(self, x):
        neighbor_matrix = x[self.neighbor_list]
        neighbor_attention_res = self._neighbor_attention(neighbor_matrix)
        x = self._attention_layer(x, neighbor_attention_res)
        
        x = F.normalize(x)
        return x

import torch
import torch.nn.functional as F
def to_one(matirx):
    return torch.where(matirx > 0, torch.ones_like(matirx), torch.zeros_like(matirx))

def padding_job(features,adj,adj_i):
    features = F.pad(features, (0, 0, 0, 1), value=0)
    adj = F.pad(adj, (0, 1, 0, 1))
    adj_i = F.pad(adj_i, (0, 1, 0, 1))
    return features,adj,adj_i
def generate_neighbor_list(adj, k,limits=150):
    adj_list = []
    adj = to_one(adj)
    adj_list.append(adj)
    hop_adj = adj
    for i in range(k-1):
        hop_adj = to_one(torch.mm(hop_adj, adj)-torch.eye(adj.shape[0])*1000)
        adj_list.append(hop_adj)

    for i in range(k-1,0,-1):
        adj_list[i]=adj_list[i]-adj_list[i-1]
    
    res=[]
    for i in range(k):
        res.append(generate_neigbor(adj_list[i],limits-30*i))
    return res
def generate_neigbor(adj,limits=150):
    '''
    adj is after padding
    '''
    max_neigbor = 0
    neigbor_list = []
    for i in range(adj.shape[0]-1):
        nonzero = list(torch.nonzero(adj[i]).reshape(-1))
        nonzero = list(map(lambda x: int(x), nonzero))
        if len(nonzero)>limits:
            nonzero=nonzero[:limits]
        max_neigbor = max(max_neigbor, len(nonzero))
        neigbor_list.append(nonzero)

    for i in range(adj.shape[0]-1):
        for j in range(max_neigbor-len(neigbor_list[i])):
            neigbor_list[i].append(adj.shape[0]-1)

    neigbor_list = torch.tensor(neigbor_list)
    neigbor_list = F.pad(neigbor_list, (0, 0, 0, 1),value=adj.shape[0]-1)
    return neigbor_list
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

    
class RC_STML(nn.Module):
    def __init__(self, sigma=1, delta=1, view=2, disable_mu=1, topk=10):
        super(RC_STML, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk
        
    def k_reciprocal_neigh(self, initial_rank, i, topk):
        forward_k_neigh_index = initial_rank[i,:topk]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:topk]
        fi = np.where(backward_k_neigh_index==i)[0]
        return forward_k_neigh_index[fi]

    def forward(self, s_emb, t_emb, idx, classes_labels=None, v2=False):
        if v2:
            return self.forward_v2(s_emb, t_emb, idx)
        # if self.disable_mu:
        #     s_emb = F.normalize(s_emb)
        t_emb = F.normalize(t_emb)

        N = len(s_emb)        
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)
        
        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb) 
            W_P = torch.exp(-T_dist.pow(2) / self.sigma)
            
            batch_size = len(s_emb) // self.view
            W_P_copy = W_P.clone()
            W_P_copy[idx.unsqueeze(1) == idx.unsqueeze(1).t()] = 1

            topk_index = torch.topk(W_P_copy, self.topk)[1]
            topk_half_index = topk_index[:, :int(np.around(self.topk/2))]

            W_NN = torch.zeros_like(W_P).scatter_(1, topk_index, torch.ones_like(W_P))
            V = ((W_NN + W_NN.t())/2 == 1).float()

            W_C_tilda = torch.zeros_like(W_P)
            for i in range(N):
                indNonzero = torch.where(V[i, :]!=0)[0]
                W_C_tilda[i, indNonzero] = (V[:,indNonzero].sum(1) / len(indNonzero))[indNonzero]
                
            W_C_hat = W_C_tilda[topk_half_index].mean(1)
            W_C = (W_C_hat + W_C_hat.t())/2
            W = (W_P + W_C)/2

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)
        
        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight
        
        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb)-1))
        
        return loss
    
    def forward_v2(self, probs, feats, idx):
        with torch.no_grad():
            pseudo_labels = probs.argmax(1).cuda()
            one_hot = torch.zeros(probs.shape).cuda().scatter(1, pseudo_labels.unsqueeze(1), 1.0)
            W_P = torch.mm(one_hot, one_hot.t())
            feats_dist = torch.cdist(feats, feats)
            topk_index = torch.topk(feats_dist, self.topk)[1]
            W_NN = torch.zeros_like(feats_dist).scatter_(1, topk_index, W_P)
            
            W = ((W_NN + W_NN.t())/2 == 0.5).float()
            
            N = len(probs)
            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)
            
        pull_losses = torch.relu(feats_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - feats_dist).pow(2) * neg_weight
        
        loss = (pull_losses.sum() + push_losses.sum()) / (len(probs) * (len(probs)-1))
        
        return loss
        
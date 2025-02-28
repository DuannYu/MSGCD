import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch_clustering

class Kmeans_Loss(nn.Module):
    def __init__(self, temperature=0.5, n_clusters=196):
        super(Kmeans_Loss, self).__init__()
        self.temperature = temperature
        self.num_cluster = n_clusters
        self.clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, n_clusters=self.num_cluster)
        self.psedo_labels = None
    
    def clustering(self, features, n_clusters):

        # kwargs = {
        #     'metric': 'cosine' if self.l2_normalize else 'euclidean',
        #     'distributed': True,
        #     'random_state': 0,
        #     'n_clusters': n_clusters,
        #     'verbose': True
        # }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, n_clusters=self.num_cluster)

        psedo_labels = clustering_model.fit_predict(features)
        self.psedo_labels = psedo_labels
        cluster_centers = clustering_model.cluster_centers_
        return psedo_labels, cluster_centers

    def compute_cluster_loss(self,
                            q_centers,
                            k_centers,
                            temperature=0.5,
                            psedo_labels=None):
        d_q = q_centers.mm(q_centers.T) / temperature
        d_k = (q_centers * k_centers).sum(dim=1) / temperature
        d_q = d_q.float()
        d_q[torch.arange(self.num_cluster), torch.arange(self.num_cluster)] = d_k

        # q -> k
        # d_q = q_centers.mm(k_centers.T) / temperature

        zero_classes = torch.arange(self.num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                                 self.num_cluster), dim=0) == 0]
        mask = torch.zeros((self.num_cluster, self.num_cluster), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_cluster, self.num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_cluster - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_cluster, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.num_cluster - len(zero_classes))
        return loss
    
    
    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.num_cluster, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers
    
    def forward(self, im_q, im_k, psedo_labels):
        batch_all_psedo_labels = psedo_labels
        q_centers = self.compute_centers(im_q, batch_all_psedo_labels)
        k_centers = self.compute_centers(im_k, batch_all_psedo_labels)
        
        cluster_loss = self.compute_cluster_loss(q_centers, k_centers, self.temperature, batch_all_psedo_labels)
        
        return cluster_loss
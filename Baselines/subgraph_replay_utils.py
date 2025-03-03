import random
import time
import numpy as np
import torch
import torch as th
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.base import DGLError
import dgl.function as fn
import copy
import torch.nn.functional as F


class my_sampler(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.t =0

    def forward(self, ids_per_cls_train, budget, reps):
        '''
            ids_per_cls_train: all training ids in each class of the current task
            budget: budget_size
        '''

        reps = reps[:,:self.t + len(ids_per_cls_train)]
        
        reps = F.softmax(reps, dim=1)

        all_ids = []
        id_range_per_cls = [0]
        for i,ids in enumerate(ids_per_cls_train):
            all_ids.extend(ids)
            id_range_per_cls.append(id_range_per_cls[i]+len(ids))

        id_range_per_cls.append(reps.shape[0])



        num_cls = len(ids_per_cls_train)
        ids_selected = [[] for i in range(num_cls)]
        for i in range(num_cls):
            low_index, high_index = id_range_per_cls[i], id_range_per_cls[i+1]
            id = reps[low_index:high_index, self.t + i].argmax()+low_index
            ids_selected[i].extend([id])

        self.t += len(ids_per_cls_train)


        min_dists = torch.ones(num_cls, reps.shape[0]).to(reps.device)*1000
        for i in range(num_cls):
            min_dists[i] = torch.min(min_dists[i], self.euclidean_similarity(reps[ids_selected[i][0]], reps))
            min_dists[i][ids_selected[i][0]] = -1000

        budget_size = [int(budget) for ids in ids_per_cls_train]

        for i in range(1, budget):
            for j in range(num_cls):
                if len(ids_selected[j]) >= min(len(ids_per_cls_train[j]),budget_size[j]):
                    continue
                low_index, high_index = id_range_per_cls[j], id_range_per_cls[j+1]
                dist = min_dists[j] + 2*(torch.sum(min_dists, dim=0)-min_dists[j])
                dist[min_dists[j]<-99] = -1000
                ids_selected[j].append(dist[low_index:high_index].argmax()+low_index)
                min_dists[j] = torch.min(min_dists[j], self.euclidean_similarity(reps[ids_selected[j][-1]], reps))
                min_dists[j][torch.tensor(ids_selected[j])] = -1000

        ids_selected = [all_ids[i] for sublist in ids_selected for i in sublist]

        return ids_selected

    
    def cosine_similarity(self, vec, mat):
        vec = vec.unsqueeze(0)
        return -torch.cosine_similarity(vec, mat, dim=1)/2+0.5
    
    def euclidean_similarity(self, vec, mat):
        vec = vec.unsqueeze(0)
        return torch.cdist(vec, mat, p=2).squeeze(0)
    
    def pairwise_euclidean_distance(self, x):
        # x should be a 2D tensor, shape: (batch_size, dim)
        # pairwise euclidean distance is calculated
        square = torch.sum(x**2, dim=1, keepdim=True)
        distance_square = -2 * torch.matmul(x, x.t()) + square + square.t()
        distance = torch.sqrt(distance_square + 1e-7) # add a small number to prevent numerical instability
        return distance

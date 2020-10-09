'''
pytorch version of NGCF
@author zhuchen  zhuchen@my.swjtu.edu.cn 2020/3/7
'''
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np
import os
import sys
from data_process import *

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, norm_adj, batch_size, decay):
        super().__init__()
        self.name = 'NGCF'
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.batch_size = batch_size
        self.norm_adj = norm_adj
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float()
        self.norm_adj = self.norm_adj.cuda()
        self.decay = decay
        self.u_f_embeddings = None   #最终得到的嵌入特征
        self.i_f_embeddings = None
        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))
        '''
        for i in range(self.n_layers):
            nn.init.xavier_uniform_(self.GC_Linear_list[i].weight)
            nn.init.zeros_(self.GC_Linear_list[i].bias)
            nn.init.xavier_uniform_(self.Bi_Linear_list[i].weight)
            nn.init.zeros_(self.Bi_Linear_list[i].bias)
        '''
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings
    
    def predict(self, userIDx, itemIDx):
        with torch.no_grad():
            if self.u_f_embeddings is not None:
                uEmbd, iEmbd = self.u_f_embeddings, self.i_f_embeddings
            else:
                uEmbd, iEmbd = self.forward(self.norm_adj)
                self.u_f_embeddings, self.i_f_embeddings = uEmbd, iEmbd
            uembd = uEmbd[userIDx]
            iembd = iEmbd[itemIDx]
            prediction = torch.sum(torch.mul(uembd,iembd),dim=1)
        return prediction


    def   sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def BPR_loss(self, users, pos_item, neg_item):
        uEmbd, iEmbd = self.forward(self.norm_adj)
        self.u_f_embeddings, self.i_f_embeddings = uEmbd, iEmbd
        uembd = uEmbd[users]
        pos_iembd = iEmbd[pos_item]
        neg_iembd = iEmbd[neg_item]

        pos_score = torch.sum(torch.mul(uembd,pos_iembd),dim=1) 
        neg_score = torch.sum(torch.mul(uembd,neg_iembd),dim=1) 
        regularizer = torch.sum(uembd**2)/2.0 + torch.sum(pos_iembd**2)/2.0 + torch.sum(neg_iembd**2)/2.0 
        # 参照NGCF loss的设置
        maxi = torch.log(torch.sigmoid(pos_score - neg_score))
        
        mf_loss = torch.mean(maxi) * -1.0
        #print('mf_loss', mf_loss.item())
        reg_loss = self.decay * regularizer / self.batch_size
        #print('reg_loss', reg_loss.item())
        return mf_loss, reg_loss
from NGCFModel import *
from data_process import *
import torch
from torch import nn as nn
from scipy.sparse import coo_matrix
import numpy as np
from numpy import diag
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from metrics import *
import os
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


batch_size = 4096
BATCH_SIZE = batch_size
epochs = 400
para = {
    'lr':0.0001,
    'train':0.8
}


def test(model, users_list):
    all_precision_20, all_recall_20, all_precision_10, all_recall_10 = [], [], [], []
    count = 0
    for j in range(BATCH_SIZE):
        id = users_list[np.random.randint(data.n_users, size=1)[0]]
        item_list = list(set(range(data.n_items)) - set(data.train_items[id]))
        #print(len(item_list))
        users = [id for j in range(len(item_list))]
        users = torch.tensor(users).cuda()
        items = torch.tensor(item_list).cuda()
        pred = model.predict(users, items)
        #pred = torch.squeeze(pred)
        _, item_key = pred.sort(descending = True)
        item_key = item_key.cpu().int()
        item_top20 = item_key[:20]
        item_top10 = item_key[:10]
        item_list = np.array(item_list)
        pred_top20 = item_list[item_top20]
        pred_top10 = item_list[item_top10]
        actual = data.test_set[id]
        precision_20 = precisionk(actual, pred_top20)
        recall_20 = recallk(actual, pred_top20)
        all_precision_20.append(precision_20)
        all_recall_20.append(recall_20)

        precision_10 = precisionk(actual, pred_top10)
        recall_10 = recallk(actual, pred_top10)
        all_precision_10.append(precision_10)
        all_recall_10.append(recall_10)

    return np.mean(all_precision_20), np.mean(all_recall_20), np.mean(all_precision_10), np.mean(all_recall_10)

def main():
    model = NGCF(user_nums, item_nums, 64, [64, 64, 64], [0.1, 0.1, 0.1], norm_adj, 4096, 0.00001).cuda()
    optim = Adam(model.parameters(), lr=para['lr'])
    #optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
    lossfn = model.BPR_loss
    print('start training')
    
    
    
    for i in range(epochs): #n_epochs // batch_size
        loss_value = 0
        mf_loss_value, reg_loss_value = 0.0, 0.0
        t0 = time()
        for j in range(n_epochs//batch_size +1):
            users, pos_items, neg_items = data.sample()
            users = torch.tensor(users).cuda()
            pos_items = torch.tensor(pos_items).cuda()
            neg_items = torch.from_numpy(np.array(neg_items)).cuda()
            optim.zero_grad()
            mf_loss, reg_loss = lossfn(users, pos_items, neg_items)
            loss = mf_loss + reg_loss
            loss.backward()
            optim.step()
            loss_value += loss.item()
            mf_loss_value += mf_loss.item()
            reg_loss_value += reg_loss.item()
        
        
        if (i+1)%10 !=0:
            str1 = 'epoch: %d %.5f=%.2f+%.5f' % (i, loss_value, mf_loss_value, reg_loss_value)
            print(str1)
            continue
        
        t1 = time()
        print('one epoch consume %.2f s' % (t1-t0))
        user_to_test = list(data.test_set.keys())
        precision_20, recall_20, precision_10, recall_10 = test(model, user_to_test)
        t2 = time()
        print('test time %.2f s' % (t2-t1))
        str1 = 'epoch: %d %.2f=%.2f+%.2f %.5f %.5f %.5f %.5f' % (i, loss_value, mf_loss_value, reg_loss_value, precision_20, recall_20, precision_10, recall_10)
        print(str1)
        
    
    all_precision_10, all_recall_10 = [], []
    all_precision_20, all_recall_20 = [], []
    all_ndcg_20 = []
    pred20 = np.zeros((user_nums, 20))
    for i in range(user_nums):
        item_list = list(set(range(data.n_users)) - set(data.train_items[i]))
        #print(len(item_list))
        users = [i for j in range(len(item_list))]
        users = torch.tensor(users).cuda()
        items = torch.tensor(item_list).cuda()
        pred = model.predict(users, items)
        #pred = torch.squeeze(pred)
        _, item_key = pred.sort(descending = True)
        item_key = item_key.cpu().int()
        item_top10 = item_key[:10]
        item_top20 = item_key[:20]
        item_list = np.array(item_list)
        pred_top10 = item_list[item_top10]
        pred_top20 = item_list[item_top20]
        pred20[i] = np.array(pred_top20)
        actual = data.test_set[i]
        precision_10 = precisionk(actual, pred_top10)
        recall_10 = recallk(actual, pred_top10)
        all_precision_10.append(precision_10)
        all_recall_10.append(recall_10)
        precision_20 = precisionk(actual, pred_top20)
        recall_20 = recallk(actual, pred_top20)
        ndcg_20 = ndcgk(actual, pred_top20, 20)
        all_precision_20.append(precision_20)
        all_recall_20.append(recall_20)
        all_ndcg_20.append(ndcg_20)
        print('recall_20: %.5f ndcg_20: %.5f' % (recall_20, ndcg_20))
        print('uid %d: pre@10 %.5f recall@10 %.5f, pre@20 %.5f recall@20 %.5f ndcg@20 %.5f' % (i, np.mean(all_precision_10), np.mean(all_recall_10),\
            np.mean(all_precision_20), np.mean(all_recall_20), np.mean(all_ndcg_20)))
 


if __name__ == '__main__':
    cur_dir = os.getcwd()  
    path = './data/Gowalla_make'
    data = Data(path, batch_size)
    #data.negative_pool()
    user_nums, item_nums = data.get_num_users_items()
    USER_NUM, ITEM_NUM = user_nums, item_nums
    n_epochs = data.get_trainNum()
    plain_adj, norm_adj, mean_adj = data.get_adj_mat()   # 返回的是拉普拉斯矩阵的三个形式， 具体看source code
    
    main()
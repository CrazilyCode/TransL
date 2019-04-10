from model import *
from data import *

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim

from _collections import defaultdict
import numpy as np
import random


class TrainDataset(Dataset):

    def __init__(self, file_name, neg_type):
        self.entity2id, self.id2entity = load_dict('../' + file_name + '/entity2id.txt')
        self.rel2id, self.id2rel = load_dict('../' + file_name + '/relation2id.txt')

        self.entity2data, self.rel2entity_list = load_data('../' + file_name + '/train.txt', self.entity2id, self.rel2id)
        
        if neg_type == 'unif':
            self.rel2entity = defaultdict(list)
            for rel in self.rel2entity_list:
                entity_set = set(self.rel2entity_list[rel])
                self.rel2entity[rel] = list(entity_set)
        elif neg_type == 'bern':
            self.rel2entity = self.rel2entity_list
        elif neg_type == 'all':
            self.rel2entity = defaultdict(list)
            for rel in self.rel2entity_list:
                self.rel2entity[rel] = list(self.id2entity.keys())

        self.id2e1_train, self.id2e2_train, self.id2rel_train = load_train_data('../' + file_name + '/train.txt')
        self.len = len(self.id2e1_train)

        self.entity_len = len(self.entity2id)
        self.rel_len = len(self.rel2id)

        self.get_data()

    def get_data(self):
        self.id2hr_train = []
        self.id2he_train = []
        self.id2h_train = []
        self.id2r_train = []
        self.id2t_train = []
        for index in range(self.len):
            e1 = self.entity2id[self.id2e1_train[index]]
            e2 = self.entity2id[self.id2e2_train[index]]
            rel = self.rel2id[self.id2rel_train[index]]

            data = self.entity2data[e1]
            data.remove((rel, e2))
            data.append((self.rel_len, e1))
            r_list = []
            e_list = []
            for (r, e) in data:
                r_list.append(rel * (self.rel_len + 1) + r)
                e_list.append(e)

            self.id2hr_train.append(r_list)
            self.id2he_train.append(e_list)
            self.id2h_train.append(e1)
            self.id2r_train.append(rel)
            self.id2t_train.append(e2)

    def get_neg(self, e1, rel, e2):
        data = self.entity2data[e1]
        right_list = []
        for (r, e) in data:
            if r == rel:
                right_list.append(e)

        while(True):
            neg_id = random.choice(self.rel2entity[rel])
            if neg_id not in right_list:
                break

        return neg_id

    def __getitem__(self, index):
        r_list_o = self.id2hr_train[index]
        e_list_o = self.id2he_train[index]
        e1 = self.id2h_train[index]
        rel = self.id2r_train[index]
        e2 = self.id2t_train[index]

        data_len_o = len(r_list_o)

        if data_len_o > 50:
            ids = random.sample(range(0, data_len_o), 50)
            r_list = [r_list_o[idx] for idx in ids]
            e_list = [e_list_o[idx] for idx in ids]
        else:
            r_list = r_list_o
            e_list = e_list_o

        data_len = len(r_list)
        data_r = torch.from_numpy(np.array(r_list))
        data_r_temp = torch.ones((50 - data_len), dtype=torch.int) * (self.rel_len * (self.rel_len + 1))
        data_r = torch.cat((data_r, data_r_temp), 0)
        
        data_e = torch.from_numpy(np.array(e_list))
        data_e_temp = torch.ones((50 - data_len), dtype=torch.int) * self.entity_len
        data_e = torch.cat((data_e, data_e_temp), 0)
        
        pos_id = e2
        neg_id = self.get_neg(e1, rel, e2)

        return data_r, data_e, rel, pos_id, neg_id

    def __len__(self):
        return self.len


def train(entity_dim, margin1, margin2, nn_lr, batch_size, train_dataset, file_name, neg_type):
    train_epochs = 500

    file_dir = file_name + '/' + str(entity_dim) + '-' + str(margin1) + '-' + str(margin2) + '(' + str(nn_lr) + '-' + str(batch_size) + ')-' + neg_type
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=5, batch_size=batch_size)
    train_len = train_dataset.len

    net = Network(train_dataset.entity_len, entity_dim, train_dataset.rel_len)
    # net = torch.load('out/' + file_dir + '/net-' + str(70) + '.pt')
    loss_func = ContrastiveLoss(margin1, margin2)
    optimizer = optim.Adam(net.parameters(), lr=nn_lr)

    torch.backends.cudnn.benchmart = True

    if torch.cuda.is_available() == True:
        net = net.cuda()
        loss_func = loss_func.cuda()
    
    start = time.time()

    for epoch in range(train_epochs):
        epoch_loss = 0
        current_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            data_r, data_e, rel, pos_id, neg_id = data

            if torch.cuda.is_available() == True:
                data_r = data_r.cuda()
                data_e = data_e.cuda()
                rel = rel.cuda()
                pos_id = pos_id.cuda()
                neg_id = neg_id.cuda()

            out_model, pos_out, neg_out = net(data_r, data_e, rel, pos_id, neg_id)

            optimizer.zero_grad()
            loss = loss_func(out_model, pos_out, neg_out)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available() == True:
                loss = loss.cpu()

            current_loss += loss.data.item()
            epoch_loss += loss.data.item()

            j = i * batch_size
            if j % 20000 == 0:
                print('%d %d %d%% (%s) %.4f' % (epoch, j, j * 100 / train_len, timeSince(start), current_loss))
                current_loss = 0
            

        if not os.path.exists('out/' + file_dir):
            os.makedirs('out/' + file_dir)
        
        j = epoch + 1
        if j % 10 == 0:
            model_name = 'out/' + file_dir + '/net-' + str(j) + '.pt'
            torch.save(net, model_name)

        loss_str = '%.4f' % epoch_loss
        write('out/' + file_dir + '/loss.txt', loss_str)

    print('train done!')

    

if __name__ == '__main__':
    entity_dim = 10

    margin1 = 1
    margin2 = 50

    nn_lr = 0.005
    batch_size = 100

    neg_type = 'unif'
    # neg_type = 'all'
    # neg_type = 'bern'

    # file_name = 'FB13'
    # file_name = 'WN11'
    file_name = 'FB15k'
    train_dataset = TrainDataset(file_name, neg_type)
    train(entity_dim, margin1, margin2, nn_lr, batch_size, train_dataset, file_name, neg_type)
    

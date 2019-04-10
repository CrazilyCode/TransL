from model import *
from data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random


class TestDataset(Dataset):

    def __init__(self, file_name, test_name):
        self.entity2id, self.id2entity = load_dict('../' + file_name + '/entity2id.txt')
        self.rel2id, self.id2rel = load_dict('../' + file_name + '/relation2id.txt')

        self.entity2data, self.rel2entity = load_data('../' + file_name + '/train.txt', self.entity2id, self.rel2id)

        self.entity_len = len(self.entity2id)
        self.rel_len = len(self.rel2id)

        self.id2e1_test, self.id2e2_test, self.id2rel_test, self.id2flag_test = load_test_data('../' + file_name + '/' + test_name)
        self.len = len(self.id2e1_test)


    def __getitem__(self, index):
        e1 = self.entity2id[self.id2e1_test[index]]
        e2 = self.entity2id[self.id2e2_test[index]]
        rel = self.rel2id[self.id2rel_test[index]]
        flag = self.id2flag_test[index]


        data = self.entity2data[e1]
        # data.remove((rel, e2))
        data.append((self.rel_len, e1))
        r_list_o = []
        e_list_o = []
        for (r, e) in data:
            r_list_o.append(rel * (self.rel_len + 1) + r)
            e_list_o.append(e)

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

        flag = int(flag)

        return data_r, data_e, rel, e2, flag

    def __len__(self):
        return self.len

def test(file_name, net_path, test_name, out_path):
    net = torch.load(net_path)

    test_dataset = TestDataset(file_name, test_name)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=1, batch_size=1000)


    dict_right1 = defaultdict(int)
    dict_right2 = defaultdict(int)
    dict_count = defaultdict(int)
    right1 = 0
    right2 = 0
    count = 0

    pdist = nn.PairwiseDistance(p=2)

    print(margin_dict)

    for i, data in enumerate(test_dataloader, 0):
        data_r, data_e, rel, t, flag = data
        if torch.cuda.is_available() == True:
            data_r = data_r.cuda()
            data_e = data_e.cuda()
            rel = rel.cuda()
            t = t.cuda()
            flag = flag.cuda()

        out_t = net.get_t(data_r, data_e, rel)
        t = net.eneity_embedding(t.long())

        dist = pdist(out_t, t)

        if torch.cuda.is_available() == True:
            rel = rel.cpu().numpy()

        data_len = t.size(0)
        for idx in range(data_len): 
            r = rel[idx]
            r_str = test_dataset.id2rel[r]  
            dict_count[r] += 1
            if dist[idx] < margin_dict[r_str] and flag[idx] == 1:
                right1 += 1
                dict_right1[r] += 1
            if dist[idx] >= margin_dict[r_str] and flag[idx] == -1:
                right2 += 1
                dict_right2[r] += 1
        
        count += data_len
        if count % 10000 == 0:
            line = '%d %d %d : %.4f' % (right1, right2, count, (right1 + right2 ) / count)
            print(line)
    line = '%d : %d %d %d : %.4f' % (epoch, right1, right2, count, (right1 + right2 ) / count)
    print(line)
    write(out_path, line)
    for rel in test_dataset.id2rel:
        if dict_count[rel] == 0:
            continue
        line = '%d: %d %d %d : %.4f' % (rel, dict_right1[rel], dict_right2[rel], dict_count[rel], (dict_right1[rel] + dict_right2[rel] ) / dict_count[rel])
        print(line)
        write(out_path, line)

if __name__ == '__main__':
    for i in range(46, 47):
        epoch = i * 10
        file_name = 'FB13'
        net_name = '50-1-10(0.005-100)-unif'

        test_name = 'test.txt'
        # test_name = 'valid.txt'

        net_path = 'out/' + file_name + '/' + net_name + '/net-' + str(epoch) + '.pt'
        margin_path = 'out/' + file_name + '/' + net_name + '/margin-' + str(epoch) + '.txt'
        out_path = 'out/' + file_name + '/' + net_name + '/test-final-' + str(epoch) + '.txt'
        # out_path = 'out/' + file_name + '/' + net_name + '/valid-final-' + str(epoch) + '.txt'


        margin_dict, _ = load_dict(margin_path)
        
        test(file_name, net_path, test_name, out_path)
    

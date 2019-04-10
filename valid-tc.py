from model import *
from data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random

import matplotlib.pyplot as plt

class ValidDataset(Dataset):

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

def valid(file_name, net_path, test_name, start, end):
    net = torch.load(net_path)

    valid_dataset = ValidDataset(file_name, test_name)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, num_workers=1, batch_size=1000)

    pdist = nn.PairwiseDistance(p=2)

    out_dict = defaultdict(list)

    for i, data in enumerate(valid_dataloader, 0):
        data_r, data_e, rel, t, flag = data
        if torch.cuda.is_available() == True:
            data_r = data_r.cuda()
            data_e = data_e.cuda()
            rel = rel.cuda()
            t = t.cuda()

        out_t = net.get_t(data_r, data_e, rel)
        t = net.eneity_embedding(t.long())

        dist = pdist(out_t, t)

        if torch.cuda.is_available() == True:
            dist = dist.cpu().data.numpy()
            rel = rel.cpu().numpy()
        flag = flag.numpy()

        data_len = t.size(0)
        for idx in range(data_len): 
            out_dict[(rel[idx], flag[idx])].append(dist[idx])

    data_dict = defaultdict(list)
    for r in valid_dataset.id2rel:
        pos_list = out_dict[(r, 1)]
        neg_list = out_dict[(r, -1)]
        rel_name = valid_dataset.id2rel[r]
        print(r)
        for margin in range(start, end):
            pos_count = 0
            neg_count = 0
            for pos in pos_list:
                if pos < margin:
                    pos_count += 1
            for neg in neg_list:
                if neg >= margin:
                    neg_count += 1
            if len(pos_list) + len(neg_list) == 0:
                break
            acc = '%.4f' % ((pos_count + neg_count) / (len(pos_list) + len(neg_list)))
            acc = float(acc)
            # print('%d : %d: %d %d : %.2f %.2f %.3f' % (margin, len(pos_list), pos_count, neg_count, pos_count/len(pos_list), neg_count/len(pos_list), (pos_count+neg_count)/(len(pos_list)*2)))
            print('%d\t%.4f' % (margin, acc))
            data_dict[rel_name].append(acc)
        print('*' * 20)

    return list(valid_dataset.rel2id.keys()), data_dict



def linePlots(rel_list, data_dict, start, end):
    x_axis = np.array([x for x in range(start, end)])
    
    # ['_has_instance', '_similar_to', '_member_meronym', '_domain_region', '_subordinate_instance_of', 
    # '_domain_topic', '_member_holonym', '_synset_domain_topic', '_has_part', '_part_of', '_type_of']

    plt.figure()
    handles = []
    for rel in rel_list:
        handle, = plt.plot(x_axis, data_dict[rel],'-o', label=rel)
        handles.append(handle)
    print(handles)
    print(rel_list)
    plt.legend(handles, fontsize=12)
    # plt.legend(handles, ['_has_instance', '_similar_to', '_member_meronym', '_domain_region', '_subordinate_instance_of', '_domain_topic', '_member_holonym', '_synset_domain_topic', '_has_part', '_part_of', '_type_of'])

    plt.xlabel('margin', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    # plt.title('WN11')
    plt.show()

if __name__ == '__main__':
    for i in range(46, 47):
        epoch = i * 10
        file_name = 'FB13'
        net_name = '50-1-10(0.005-100)-unif'

        test_name = 'valid.txt'
        start = 0
        end = 20

        net_path = 'out/' + file_name + '/' + net_name + '/net-' + str(epoch) + '.pt'
        out_path = 'out/' + file_name + '/' + net_name + '/margin-' + str(epoch) + '.txt'
        
        rel_list, data_dict = valid(file_name, net_path, test_name, start, end)

        print(rel_list)
        print(data_dict)

        margin_list = np.arange(start, end)
        for rel in rel_list:
            if rel not in data_dict.keys():
                continue
            data = data_dict[rel]
            index = np.argmax(data)
            margin = margin_list[index]

            line = '%s\t%d' % (rel, margin)
            write(out_path, line)

        # linePlots(rel_list, data_dict, start, end)
    

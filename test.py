# from pandas import read_pickle
import torch
import numpy as np
import os

load_npy_path = 'C:/Users/lab/Desktop/taskey/dgl/ntu'
name_list = os.listdir(load_npy_path)

# data_list = np.array(())
data_list = []
label_list = []
for idx, name in enumerate(name_list):
    if int(name[17:20]) <= 41:
        data = np.load(load_npy_path + '/' +name).item()
        skeleton2d = data['rgb_body0'].tolist()
        label = int(name[17:20])
        
        # data_list = np.append(data_list, skeleton2d[10:74])
        data_list.append(skeleton2d[0:32])
        label_list.append(label)


train_data = np.array(data_list[0:int(len(label_list)*0.8)])
train_label = label_list[0:int(len(label_list)*0.8)]

valid_data = np.array(data_list[int(len(label_list)*0.8):-1])
valid_label = label_list[int(len(label_list)*0.8):-1]

# train_label = torch.from_numpy(np.asarray(train_label))

torch.save((train_data, train_label), 'data/train.pkl')
torch.save((valid_data, valid_label), 'data/valid.pkl')


train1_tensor, train1_label = torch.load('data/train.pkl')
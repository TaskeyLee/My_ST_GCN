import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
from graph_generator import generate_graph
from tensorboardX import SummaryWriter
import Cartesian2Polar 
import codecs
import csv

def dataloader():
    # 读入数据集
    train1_tensor, train1_label = torch.load('../dataset/train.pkl')
    train2_tensor, train2_label = torch.load('../dataset/valid.pkl')
    
    train_label = torch.cat([train1_label, train2_label], dim = 0)
    
    valid_tensor , valid_label  = torch.load('../dataset/test.pkl')
    
    
    # # 将世界坐标系+笛卡尔坐标系转为相对中心坐标系+极坐标系
    # coordinate_normalization = Cartesian2Polar.Cartesian2Polar(train_tensor)
    # train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Center_Polar()
    # train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)
    
    # coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
    # valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Center_Polar()
    # valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)
    
    # coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
    # test_normalized_polar_distance, test_normalized_polar_angle = coordinate_normalization.Center_Polar()
    # test_normalized_polar_distance = torch.unsqueeze(test_normalized_polar_distance, dim=3)
    
    ##############################################################################################################
    
    # # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+极坐标系
    # coordinate_normalization = Cartesian2Polar.Cartesian2Polar(train_tensor)
    # train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Relative_Polar()
    # train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)
    
    # coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
    # valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Relative_Polar()
    # valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)
    
    # coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
    # test_normalized_polar_distance, test_normalized_polar_angle = coordinate_normalization.Relative_Polar()
    # test_normalized_polar_distance = torch.unsqueeze(test_normalized_polar_distance, dim=3)
    
    #############################################################################################################
    
    # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+极坐标系
    coordinate_normalization = Cartesian2Polar.Cartesian2Polar(train1_tensor)
    train1_normalized_cartesian_location = coordinate_normalization.Relative_Cartesian()
    
    coordinate_normalization = Cartesian2Polar.Cartesian2Polar(train2_tensor)
    train2_normalized_cartesian_location = coordinate_normalization.Relative_Cartesian()
    
    train_normalized_cartesian_location = torch.cat([train1_normalized_cartesian_location, train2_normalized_cartesian_location], dim = 0)
    
    
    coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
    valid_normalized_cartesian_location = coordinate_normalization.Relative_Cartesian()
    # Dataloader使用的数据处理函数
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels
    
    
    # 将坐标转换后的数据集转为graph
    # 训练集
    data_type = 'Cartesian'
    
        
    # if data_type == 'Polar':
    #     body_graphs = []
    #     for idx, tensor in enumerate(train_normalized_polar_angle):
    #         g = generate_graph() # graph有15个node，29条edge
    #         g.ndata['angle'] = tensor.permute(1,0,2)
    #         g.ndata['distance'] = train_normalized_polar_distance[idx].permute(1,0,2)
    #         body_graphs.append([g, torch.tensor(train_label[idx])])
            
    #     train_loader = data.DataLoader(body_graphs,
    #                                     collate_fn = collate,
    #                                     batch_size = 16,
    #                                     shuffle = True)  
    #     # 验证集
    #     body_graphs = []
    #     for idx, tensor in enumerate(valid_normalized_polar_angle):
    #         g = generate_graph() # graph有15个node，29条edge
    #         g.ndata['angle'] = tensor.permute(1,0,2)
    #         g.ndata['distance'] = valid_normalized_polar_distance[idx].permute(1,0,2)
    #         body_graphs.append([g, torch.tensor(valid_label[idx])])
            
    #     valid_loader = data.DataLoader(body_graphs,
    #                                     collate_fn = collate,
    #                                     batch_size = 16,
    #                                     shuffle = True)  
       
    if data_type == 'Cartesian':
        body_graphs = []
        for idx, tensor in enumerate(train_normalized_cartesian_location):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['location'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(train_label[idx])])
            
        train_loader = data.DataLoader(body_graphs,
                                       collate_fn = collate,
                                       batch_size = 16,
                                       shuffle = True)  
        # 验证集
        body_graphs = []
        for idx, tensor in enumerate(valid_normalized_cartesian_location):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['location'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(valid_label[idx])])
            
        valid_loader = data.DataLoader(body_graphs,
                                       collate_fn = collate,
                                       batch_size = 16,
                                       shuffle = True)  
    return train_loader, valid_loader
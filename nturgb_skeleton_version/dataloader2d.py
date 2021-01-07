import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
from graph_generator import generate_graph
from tensorboardX import SummaryWriter
import coordinate_transform2d
import codecs
import csv

def dataloader(data_type):
    # 读入数据集
    train_tensor, train_label = torch.load('data/train.pkl')
    
    valid_tensor, valid_label = torch.load('data/valid.pkl')

    if data_type == 'center_polar': 
        # 将世界坐标系+笛卡尔坐标系转为相对中心坐标系+极坐标系
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(train_tensor)
        train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Center_Polar()
        train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)
        
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(valid_tensor)
        valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Center_Polar()
        valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)
    
    #############################################################################################################
    
    if data_type == 'relative_polar': 
        # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+极坐标系
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(train_tensor)
        train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Relative_Polar()
        train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)
        
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(valid_tensor)
        valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Relative_Polar()
        valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)
    
    ############################################################################################################
    
    if data_type == 'center_cartesian': 
        # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+笛卡尔坐标系
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(train_tensor)
        train_normalized_cartesian_location = coordinate_normalization.Center_Cartesian()
        
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(valid_tensor)
        valid_normalized_cartesian_location = coordinate_normalization.Center_Cartesian()
        
    if data_type == 'relative_cartesian': 
        # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+笛卡尔坐标系
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(train_tensor)
        train_normalized_cartesian_location = coordinate_normalization.Relative_Cartesian()
        
        coordinate_normalization = coordinate_transform2d.Coordinate_transform(valid_tensor)
        valid_normalized_cartesian_location = coordinate_normalization.Relative_Cartesian()
        
        
    # Dataloader使用的数据处理函数
    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels
    
    
    # 将坐标转换后的数据集转为graph
    # 训练集
    
        
    if data_type == 'center_polar' or data_type == 'relative_polar':
        body_graphs = []
        for idx, tensor in enumerate(train_normalized_polar_angle):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['angle'] = tensor.permute(1,0,2)
            g.ndata['distance'] = train_normalized_polar_distance[idx].permute(1,0,2)
            body_graphs.append([g, torch.tensor(train_label[idx])])
            
        train_loader = data.DataLoader(body_graphs,
                                        collate_fn = collate,
                                        batch_size = 32,
                                        shuffle = True)  
        # 验证集
        body_graphs = []
        for idx, tensor in enumerate(valid_normalized_polar_angle):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['angle'] = tensor.permute(1,0,2)
            g.ndata['distance'] = valid_normalized_polar_distance[idx].permute(1,0,2)
            body_graphs.append([g, torch.tensor(valid_label[idx])])
            
        valid_loader = data.DataLoader(body_graphs,
                                        collate_fn = collate,
                                        batch_size = 32,
                                        shuffle = True)  
       
    if data_type == 'center_cartesian' or data_type == 'relative_cartesian':
        body_graphs = []
        for idx, tensor in enumerate(train_normalized_cartesian_location):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['location'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(train_label[idx])])
            
        train_loader = data.DataLoader(body_graphs,
                                       collate_fn = collate,
                                       batch_size = 32,
                                       shuffle = True)  
        # 验证集
        body_graphs = []
        for idx, tensor in enumerate(valid_normalized_cartesian_location):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['location'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(valid_label[idx])])
            
        valid_loader = data.DataLoader(body_graphs,
                                       collate_fn = collate,
                                       batch_size = 32,
                                       shuffle = True)  
        
    
    
    return train_loader, valid_loader
train, valid = dataloader('center_polar')
import torch
from torch.utils import data
import numpy as np
import dgl
from graph_generator import generate_graph
import coordinate_transform
import random
import math

torch.manual_seed(999)

# 使用仿射变换实现数据增强
def data_enhance(train, train_label):
    train_enhance = [] # 存储所有视频样本
    train_label_enhance = [] # 存储所有视频label
    for idx, train_video in enumerate(train):
        # 存储一个视频中的32帧
        for i in range(3): # 每个样本进行3次数据增强
            train_video_enhance = []
            degree = random.randint(-6,6) * 10
            for frame in train_video:
                # 旋转变换矩阵
                matrix1 = np.array(((math.cos(degree*math.pi/180), math.sin(degree*math.pi/180), 0), (0, 1, 0), (0, -math.sin(10*math.pi/180), math.cos(10*math.pi/180)))) # 变换矩阵
                # 剪切变换矩阵
                matrix2 = np.array(((1, math.tan(degree*math.pi/180), 0), (0, 1, 0), (0, 0, 1))) # 变换矩阵
                # 综合变换矩阵
                matrix = np.dot(matrix1, matrix2)
                frame_enhanced = np.dot(frame, matrix) # 取出一个样本中的一帧的人体节点信息，dim=（15，3）
                train_video_enhance.append(frame_enhanced)
                
            train_label_enhance.append(train_label[idx])
            train_enhance.append(train_video_enhance)
    
    train_enhance, train_label_enhance = torch.tensor(train_enhance), torch.tensor(train_label_enhance)
    return train_enhance, train_label_enhance

def dataloader(data_type):
    # 读入数据集
    train1_tensor, train1_label = torch.load('../dataset/train.pkl')
    train2_tensor, train2_label = torch.load('../dataset/valid.pkl')
    
    train_tensor = torch.cat([train1_tensor, train2_tensor], dim = 0)
    train_label = torch.cat([train1_label, train2_label], dim = 0)
    
    # 数据增强：随机仿射变换进行数据增强
    # train_tensor, train_label = data_enhance(train_tensor, train_label)
    
    # 数据分组：随机分为3组
    # train_1, train_label_1 = train_tensor[0:int(train_tensor.size()[0] / 3)], train_label[0:int(train_tensor.size()[0] / 3)]
    # train_2, train_label_2 = train_tensor[int(train_tensor.size()[0] / 3):int(train_tensor.size()[0] / 3 * 2)], train_label[int(train_tensor.size()[0] / 3):int(train_tensor.size()[0] / 3 * 2)]
    # train_3, train_label_3 = train_tensor[int(train_tensor.size()[0] / 3 * 2):train_tensor.size()[0]], train_label[int(train_tensor.size()[0] / 3 * 2):train_tensor.size()[0]]
    
    
    
    
    valid_tensor , valid_label  = torch.load('../dataset/test.pkl')

    if data_type == 'center_polar': 
        # 将世界坐标系+笛卡尔坐标系转为相对中心坐标系+极坐标系
        coordinate_normalization = coordinate_transform.Coordinate_transform(train_tensor)
        train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Center_Polar()
        train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)
        train_normalized_polar = torch.cat((train_normalized_polar_distance, train_normalized_polar_angle), 3)
        
        coordinate_normalization = coordinate_transform.Coordinate_transform(valid_tensor)
        valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Center_Polar()
        valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)
        valid_normalized_polar = torch.cat((valid_normalized_polar_distance, valid_normalized_polar_angle), 3)
        
    #############################################################################################################
    
    if data_type == 'relative_polar': 
        # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+极坐标系
        coordinate_normalization = coordinate_transform.Coordinate_transform(train_tensor)
        train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Relative_Polar()
        train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)
        train_normalized_polar = torch.cat((train_normalized_polar_distance, train_normalized_polar_angle), 3)
        
        coordinate_normalization = coordinate_transform.Coordinate_transform(valid_tensor)
        valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Relative_Polar()
        valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)
        valid_normalized_polar = torch.cat((valid_normalized_polar_distance, valid_normalized_polar_angle), 3)
        
    ############################################################################################################
    
    if data_type == 'center_cartesian': 
        # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+笛卡尔坐标系
        coordinate_normalization = coordinate_transform.Coordinate_transform(train_tensor)
        train_normalized_cartesian_location = coordinate_normalization.Center_Cartesian()
        
        coordinate_normalization = coordinate_transform.Coordinate_transform(valid_tensor)
        valid_normalized_cartesian_location = coordinate_normalization.Center_Cartesian()
        
    if data_type == 'relative_cartesian': 
        # 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+笛卡尔坐标系
        coordinate_normalization = coordinate_transform.Coordinate_transform(train_tensor)
        train_normalized_cartesian_location = coordinate_normalization.Relative_Cartesian()
        
        coordinate_normalization = coordinate_transform.Coordinate_transform(valid_tensor)
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
        for idx, tensor in enumerate(train_normalized_polar):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['polar'] = tensor.permute(1,0,2)
            # g.ndata['distance'] = train_normalized_polar_distance[idx].permute(1,0,2)
            body_graphs.append([g, torch.tensor(train_label[idx])])
            
        train_loader = data.DataLoader(body_graphs,
                                        collate_fn = collate,
                                        batch_size = 64,
                                        shuffle = True)  
        # 验证集
        body_graphs = []
        for idx, tensor in enumerate(valid_normalized_polar):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['polar'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(valid_label[idx])])
            
        valid_loader = data.DataLoader(body_graphs,
                                        collate_fn = collate,
                                        batch_size = 64,
                                        shuffle = True)  
       
    if data_type == 'center_cartesian' or data_type == 'relative_cartesian':
        body_graphs = []
        for idx, tensor in enumerate(train_normalized_cartesian_location):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['location'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(train_label[idx])])
            
        train_loader = data.DataLoader(body_graphs,
                                       collate_fn = collate,
                                       batch_size = 64,
                                       shuffle = True)  
        # 验证集
        body_graphs = []
        for idx, tensor in enumerate(valid_normalized_cartesian_location):
            g = generate_graph() # graph有15个node，29条edge
            g.ndata['location'] = tensor.permute(1,0,2)
            body_graphs.append([g, torch.tensor(valid_label[idx])])
            
        valid_loader = data.DataLoader(body_graphs,
                                       collate_fn = collate,
                                       batch_size = 64,
                                       shuffle = True)  
        
    
    
    return train_loader, valid_loader

# train_loader, valid_loader = dataloader('center_polar')
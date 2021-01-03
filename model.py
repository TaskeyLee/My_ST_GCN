# import torch
import torch.nn as nn
import torch.nn.functional as F

# import numpy as np
import dgl
import dgl.nn.pytorch as dglnn

class ST_GCN(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, hidden_dim1, hidden_dim2, hidden_dim3, n_class, data_type): # in_dim：每个节点输入特征维度，in_dim：每个节点输出特征维度
        super(ST_GCN, self).__init__()
        self.data_type = data_type
        if data_type == 'Polar':
            self.GraphConv_1_1 = dglnn.GraphConv(in_dim_1, hidden_dim1)
            self.GraphConv_1_2 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv_1_3 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv_1_4 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv_1_5 = dglnn.GraphConv(hidden_dim1, hidden_dim2)
            self.GraphConv_1_6 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv_1_7 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv_1_8 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv_1_9 = dglnn.GraphConv(hidden_dim2, hidden_dim3)
            self.GraphConv_1_10 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.GraphConv_1_11 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.GraphConv_1_12 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            
            self.GraphConv_2_1 = dglnn.GraphConv(in_dim_2, hidden_dim1)
            self.GraphConv_2_2 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv_2_3 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv_2_4 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv_2_5 = dglnn.GraphConv(hidden_dim1, hidden_dim2)
            self.GraphConv_2_6 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv_2_7 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv_2_8 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv_2_9 = dglnn.GraphConv(hidden_dim2, hidden_dim3)
            self.GraphConv_2_10 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.GraphConv_2_11 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.GraphConv_2_12 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            
            self.fc = nn.Linear(hidden_dim3, n_class)
        if data_type == 'Cartesian':
            self.GraphConv1 = dglnn.GraphConv(in_dim_1, hidden_dim1)
            self.GraphConv2 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv3 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv4 = dglnn.GraphConv(hidden_dim1, hidden_dim1)
            self.GraphConv5 = dglnn.GraphConv(hidden_dim1, hidden_dim2)
            self.GraphConv6 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv7 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv8 = dglnn.GraphConv(hidden_dim2, hidden_dim2)
            self.GraphConv9 = dglnn.GraphConv(hidden_dim2, hidden_dim3)
            self.GraphConv10 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.GraphConv11 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.GraphConv12 = dglnn.GraphConv(hidden_dim3, hidden_dim3)
            self.fc = nn.Linear(hidden_dim3, n_class)
    def forward(self, g, inputs):
        if self.data_type == 'Polar':
            # 每层加入resnet结构
            h1 = inputs['angle'].float()
            h2 = inputs['distance'].float()
            h1 = F.relu(self.GraphConv_1_1(g, h1))
            x = h1
            h1 = F.relu(self.GraphConv_1_2(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv_1_3(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv_1_4(g, h1)) + x
            h1 = F.relu(self.GraphConv_1_5(g, h1))
            x = h1
            h1 = F.relu(self.GraphConv_1_6(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv_1_7(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv_1_8(g, h1)) + x
            h1 = F.relu(self.GraphConv_1_9(g, h1))
            x = h1
            h1 = F.relu(self.GraphConv_1_10(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv_1_11(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv_1_12(g, h1)) + x
            
            
            h2 = F.relu(self.GraphConv_2_1(g, h2))
            x = h2
            h2 = F.relu(self.GraphConv_2_2(g, h2)) + x
            x = h2
            h2 = F.relu(self.GraphConv_2_3(g, h2)) + x
            x = h2
            h2 = F.relu(self.GraphConv_2_4(g, h2)) + x
            h2 = F.relu(self.GraphConv_2_5(g, h2))
            x = h2
            h2 = F.relu(self.GraphConv_2_6(g, h2)) + x
            x = h2
            h2 = F.relu(self.GraphConv_2_7(g, h2)) + x
            x = h2
            h2 = F.relu(self.GraphConv_2_8(g, h2)) + x
            h2 = F.relu(self.GraphConv_2_9(g, h2))
            x = h2
            h2 = F.relu(self.GraphConv_2_10(g, h1)) + x
            x = h2
            h2 = F.relu(self.GraphConv_2_11(g, h2)) + x
            x = h2
            h2 = F.relu(self.GraphConv_2_12(g, h2)) + x
            with g.local_scope():
                g.ndata['feature_angle'] = h1
                g.ndata['feature_distance'] = h2
                # print('g.h size: {}'.format(h.size()))
                readout_angle = dgl.mean_nodes(g, 'feature_angle')
                readout_distance = dgl.mean_nodes(g, 'feature_distance')
                readout_angle = readout_angle.mean(dim = 1)
                readout_distance = readout_distance.mean(dim = 1)
                readout = (readout_angle + readout_distance) / 2
                
                output = F.log_softmax(self.fc(readout))
                
        if self.data_type == 'Cartesian':
            # 每层加入resnet结构
            h1 = inputs['location'].float()
            h1 = F.relu(self.GraphConv1(g, h1))
            x = h1
            h1 = F.relu(self.GraphConv2(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv3(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv4(g, h1)) + x
            h1 = F.relu(self.GraphConv5(g, h1))
            x = h1
            h1 = F.relu(self.GraphConv6(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv7(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv8(g, h1)) + x
            h1 = F.relu(self.GraphConv9(g, h1))
            x = h1
            h1 = F.relu(self.GraphConv10(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv11(g, h1)) + x
            x = h1
            h1 = F.relu(self.GraphConv12(g, h1)) + x
            with g.local_scope():
                g.ndata['feature_location'] = h1
                # print('g.h size: {}'.format(h.size()))
                readout_location = dgl.mean_nodes(g, 'feature_location')
            
                readout_location = readout_location.mean(dim = 1)
                
                output = F.log_softmax(self.fc(readout_location))
            
        return output
        
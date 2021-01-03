import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
from graph_generator import generate_graph
from tensorboardX import SummaryWriter
import Cartesian2Polar 

# 读入数据集
train_tensor, train_label = torch.load('../dataset/train.pkl')
valid_tensor, valid_label = torch.load('../dataset/valid.pkl')
test_tensor , test_label  = torch.load('../dataset/test.pkl')

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

# 将世界坐标系+笛卡尔坐标系转为相对向心节点坐标系+极坐标系
coordinate_normalization = Cartesian2Polar.Cartesian2Polar(train_tensor)
train_normalized_polar_distance, train_normalized_polar_angle = coordinate_normalization.Relative_Polar()
train_normalized_polar_distance = torch.unsqueeze(train_normalized_polar_distance, dim=3)

coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
valid_normalized_polar_distance, valid_normalized_polar_angle = coordinate_normalization.Relative_Polar()
valid_normalized_polar_distance = torch.unsqueeze(valid_normalized_polar_distance, dim=3)

coordinate_normalization = Cartesian2Polar.Cartesian2Polar(valid_tensor)
test_normalized_polar_distance, test_normalized_polar_angle = coordinate_normalization.Relative_Polar()
test_normalized_polar_distance = torch.unsqueeze(test_normalized_polar_distance, dim=3)




# Dataloader使用的数据处理函数
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels


# 将坐标转换后的数据集转为graph
# 训练集
body_graphs = []
for idx, tensor in enumerate(train_normalized_polar_angle):
    g = generate_graph() # graph有15个node，29条edge
    g.ndata['angle'] = tensor.permute(1,0,2)
    g.ndata['distance'] = train_normalized_polar_distance[idx].permute(1,0,2)
    body_graphs.append([g, torch.tensor(train_label[idx])])
    
train_loader = data.DataLoader(body_graphs,
                               collate_fn = collate,
                               batch_size = 16,
                               shuffle = True)  
# 验证集
body_graphs = []
for idx, tensor in enumerate(valid_normalized_polar_angle):
    g = generate_graph() # graph有15个node，29条edge
    g.ndata['angle'] = tensor.permute(1,0,2)
    g.ndata['distance'] = valid_normalized_polar_distance[idx].permute(1,0,2)
    body_graphs.append([g, torch.tensor(train_label[idx])])
    
valid_loader = data.DataLoader(body_graphs,
                               collate_fn = collate,
                               batch_size = 16,
                               shuffle = True)  
# 测试集 
body_graphs = []
for idx, tensor in enumerate(test_normalized_polar_angle):
    g = generate_graph() # graph有15个node，29条edge
    g.ndata['angle'] = tensor.permute(1,0,2)
    g.ndata['distance'] = test_normalized_polar_distance[idx].permute(1,0,2)
    body_graphs.append([g, torch.tensor(train_label[idx])])

test_loader = data.DataLoader(body_graphs,
                               collate_fn = collate,
                               batch_size = 16,
                               shuffle = True)  

class ST_GCN(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, hidden_dim1, hidden_dim2, hidden_dim3, n_class): # in_dim：每个节点输入特征维度，in_dim：每个节点输出特征维度
        super(ST_GCN, self).__init__()
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
    def forward(self, g, inputs):
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
            
            return output

# 模型验证函数
def evaluate(model, dataloader, epoch):
    count = 0
    for features, labels in dataloader:
        model.eval()
        with torch.no_grad():
            output = model(features, features.ndata)
            pred = torch.argmax(output, dim = 1)
            # print('Output_size: {}, Prediction_size: {}'.format(output.size(), pred.size()))print('Prediction: {}, Label: {}'.format(pred, labels))
            # print('Output: {}'.format(output))
            # print('Prediction: {}, Label: {}'.format(pred, labels))
            correct = torch.sum(pred == labels)
            # print('Correct number: {}'.format(correct))
            count += correct
    acc = count.item() * 1.0 / len(dataloader.dataset)   
    
    return acc

# g = generate_graph() # graph有15个node，29条edge
# g.ndata['feature'] = train_tensor[0].permute(1,0,2)

model = ST_GCN(2, 1, 64, 128, 256, 10)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# 统计模型参数量
num_params = 0
for p in model.parameters():
	num_params += p.numel()
print(model)
print('The number of parameters: {}'.format(num_params))

one_hot = torch.nn.functional.one_hot(torch.arange(10), 10)

writer = SummaryWriter('logs')

for epoch in range(80):
    for idx, (feats, labels) in enumerate(train_loader):
        # labels = one_hot[labels.numpy()]
        model.train()
        output = model(feats, feats.ndata)
        # print(output)
        loss = F.nll_loss(output, labels.long())
        
        train_acc = evaluate(model, train_loader, epoch)
        valid_acc = evaluate(model, valid_loader, epoch)
            
        writer.add_scalar('Train_Loss', loss, global_step = 16 * epoch + idx)
        writer.add_scalar('Train_Accuracy', train_acc, global_step = 16 * epoch + idx)
        writer.add_scalar('Valid_Accuracy', valid_acc, global_step = 16 * epoch + idx)
        print('Epoch: {}; Loss: {}'.format(epoch, loss.squeeze()))
        print('Epoch: {}; Train_Accuracy: {} %'.format(epoch, train_acc * 100))
        print('Epoch: {}; Valid_Accuracy: {} %'.format(epoch, valid_acc * 100))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
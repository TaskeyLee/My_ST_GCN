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

coordinate_normalization = Cartesian2Polar.Cartesian2Polar(train_tensor)
normalized_polar_distance = coordinate_normalization.coordinate2distance()
normalized_polar_angle = coordinate_normalization.coordinate2angle()
normalized_polar_distance = torch.unsqueeze(normalized_polar_distance, dim=3)
# normalized_polar_train_tensor = torch.cat((normalized_polar_distance, normalized_polar_angle))



# Dataloader使用的数据处理函数
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels


# 将从pkl文件读入的数据集转为graph
body_graphs = []
for idx, tensor in enumerate(train_tensor):
    g = generate_graph() # graph有15个node，29条edge
    g.ndata['feature'] = tensor.permute(1,0,2)
    body_graphs.append([g, torch.tensor(train_label[idx])])
    
train_loader = data.DataLoader(body_graphs,
                               collate_fn = collate,
                               batch_size = 16,
                               shuffle = True)  
  
body_graphs = []
for idx, tensor in enumerate(valid_tensor):
    g = generate_graph() # graph有15个node，29条edge
    g.ndata['feature'] = tensor.permute(1,0,2)
    body_graphs.append([g, torch.tensor(train_label[idx])])
    
valid_loader = data.DataLoader(body_graphs,
                               collate_fn = collate,
                               batch_size = 16,
                               shuffle = True)  
    
body_graphs = []
for idx, tensor in enumerate(test_tensor):
    g = generate_graph() # graph有15个node，29条edge
    g.ndata['feature'] = tensor.permute(1,0,2)
    body_graphs.append([g, torch.tensor(train_label[idx])])

test_loader = data.DataLoader(body_graphs,
                               collate_fn = collate,
                               batch_size = 16,
                               shuffle = True)  

class ST_GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, hidden_dim3, n_class): # in_dim：每个节点输入特征维度，in_dim：每个节点输出特征维度
        super(ST_GCN, self).__init__()
        self.GraphConv1 = dglnn.GraphConv(in_dim, hidden_dim1)
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
        # 每层加入resnet结构
        h = F.relu(self.GraphConv1(g, inputs))
        x = h
        h = F.relu(self.GraphConv2(g, h)) + x
        x = h
        h = F.relu(self.GraphConv3(g, h)) + x
        x = h
        h = F.relu(self.GraphConv4(g, h)) + x
        h = F.relu(self.GraphConv5(g, h))
        x = h
        h = F.relu(self.GraphConv6(g, h)) + x
        x = h
        h = F.relu(self.GraphConv7(g, h)) + x
        x = h
        h = F.relu(self.GraphConv8(g, h)) + x
        h = F.relu(self.GraphConv9(g, h))
        x = h
        h = F.relu(self.GraphConv10(g, h)) + x
        x = h
        h = F.relu(self.GraphConv11(g, h)) + x
        x = h
        h = F.relu(self.GraphConv12(g, h)) + x
        with g.local_scope():
            g.ndata['feature'] = h
            # print('g.h size: {}'.format(h.size()))
            readout = dgl.mean_nodes(g, 'feature')
            readout = readout.mean(dim = 1)
            output = F.log_softmax(self.fc(readout))
            
            return output

# 模型验证函数
def evaluate(model, dataloader, epoch):
    count = 0
    for features, labels in dataloader:
        model.eval()
        with torch.no_grad():
            output = model(features, features.ndata['feature'].float())
            pred = torch.argmax(output, dim = 1)
            # print('Output_size: {}, Prediction_size: {}'.format(output.size(), pred.size()))print('Prediction: {}, Label: {}'.format(pred, labels))
            # print('Output: {}'.format(output))
            # print('Prediction: {}, Label: {}'.format(pred, labels))
            correct = torch.sum(pred == labels)
            # print('Correct number: {}'.format(correct))
            count += correct
    acc = count.item() * 1.0 / len(dataloader.dataset)   
    
    return acc

g = generate_graph() # graph有15个node，29条edge
g.ndata['feature'] = train_tensor[0].permute(1,0,2)

model = ST_GCN(3, 64, 128, 256, 10)
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
        output = model(feats, feats.ndata['feature'].float())
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
        
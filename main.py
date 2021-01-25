import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import dataloader
import codecs
import csv
from model import ST_GCN
from tensorboardX import SummaryWriter
import torch.optim as optim
import time
# import generate_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 载入数据
train_loader, valid_loader = dataloader.dataloader('relative_polar')

# 权重初始化
# def weights_init(m):
#     # classname = m.__class__.__name__
#     # if classname.find('Linear') != -1:
#     init.xavier_normal_(m.weight.data)
#     init.constant_(m.bias.data, 0.0)
# 定义保存loss、acc的函数
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for d in datas:
        writer.writerow(d)
    print("保存文件成功，处理结束")
    
# 模型验证函数
def evaluate(model, data, epoch):
    count = 0
    for features, labels in data:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
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
    acc = count.item() * 1.0 / len(data.dataset)   
    
    return acc

# g = generate_graph() # graph有15个node，29条edge
# g.ndata['feature'] = train_tensor[0].permute(1,0,2)

model = ST_GCN(3, 0, 64, 128, 256, 10, 'Polar')
model.to(DEVICE)

# weights_init(model)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# 统计模型参数量
num_params = 0
for p in model.parameters():
	num_params += p.numel()
print(model)
print('The number of parameters: {}'.format(num_params))

one_hot = torch.nn.functional.one_hot(torch.arange(10), 10)

# center_polar = SummaryWriter('logs/center_polar')
relative_polar = SummaryWriter('logs/relative_polar')
# center_cartesian = SummaryWriter('logs/center_cartesian')
# relative_cartesian = SummaryWriter('logs/relative_cartesian')

# 三个空列表，存储训练过程的loss、train accuracy、valid accuracy
lossData=[[]]
train_acc_Data=[[]]
valid_acc_Data=[[]]

index = 0
step = 1

start = time.time()

for epoch in range(500):
    for idx, (feats, labels) in enumerate(train_loader):
        # labels = one_hot[labels.numpy()]
        model.train()
        
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        
        output = model(feats, feats.ndata)
        # print(output)
        loss = F.nll_loss(output, labels.long())
        
        train_acc = evaluate(model, train_loader, epoch)
        valid_acc = evaluate(model, valid_loader, epoch)
        
        lossData.append([index,loss.to('cpu').data.numpy()])
        train_acc_Data.append([index,train_acc])
        valid_acc_Data.append([index,valid_acc])
        
        relative_polar.add_scalar('Train_Loss', loss, global_step = step)
        relative_polar.add_scalar('Train_Accuracy', train_acc, global_step = step)
        relative_polar.add_scalar('Valid_Accuracy', valid_acc, global_step = step)
        print('Epoch: {}; Loss: {}'.format(epoch, loss.squeeze()))
        print('Epoch: {}; Train_Accuracy: {} %'.format(epoch, train_acc * 100))
        print('Epoch: {}; Valid_Accuracy: {} %'.format(epoch, valid_acc * 100))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        index += 1
        
        step += 1
        
        

stop = time.time()
print('Time:', stop - start)

# 将loss、acc保存为csv文件
# 1: center + polar
# 2: relative + polar
# 3: center + cartesian
# 4: relative + cartesian

# data_write_csv(".\\loss1.csv", lossData)
# data_write_csv(".\\train_acc1.csv", train_acc_Data)
# data_write_csv(".\\valid_acc1.csv", valid_acc_Data)

data_write_csv(".\\loss2.csv", lossData)
data_write_csv(".\\train_acc2.csv", train_acc_Data)
data_write_csv(".\\valid_acc2.csv", valid_acc_Data)

# data_write_csv(".\\loss3.csv", lossData)
# data_write_csv(".\\train_acc3.csv", train_acc_Data)
# data_write_csv(".\\valid_acc3.csv", valid_acc_Data)

# data_write_csv(".\\loss4.csv", lossData)
# data_write_csv(".\\train_acc4.csv", train_acc_Data)
# data_write_csv(".\\valid_acc4.csv", valid_acc_Data)

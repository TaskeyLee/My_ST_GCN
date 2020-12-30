import numpy as np
import dgl
import torch
# graph中src和dst节点(人体关键点指向关系)

def generate_graph():
    # 建立有向图,指向向心节点
    src = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2])
    dst = np.array([1, 1, 3, 4, 1, 6, 7, 2, 9, 10, 2, 12, 13, 1])
    
    # # 建立无向图
    # src = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 1, 1, 3, 4, 1, 6, 7, 2, 9, 10, 2, 12, 13, 1])
    # dst = np.array([1, 1, 3, 4, 1, 6, 7, 2, 9, 10, 2, 12, 13, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2])
    
    g = dgl.DGLGraph((src, dst)) # 建立graph
    g = dgl.add_self_loop(g)
    
    # train_tensor, train_label = torch.load('dataset/train.pkl')
    # valid_tensor, valid_label = torch.load('dataset/valid.pkl')
    # test_tensor , test_label  = torch.load('dataset/test.pkl')
    
    # 为graph节点特征赋值，dim=（15，32，3）
    # feat1 = train_tensor[0]
    # g.ndata['feat'] = feat1.permute(1,0,2)
    return g
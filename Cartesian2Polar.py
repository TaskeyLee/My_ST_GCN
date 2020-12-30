import torch
import numpy as np

# Head：0
# Neck：1
# Spine：2
# Left Shoulder: 3
# Left Elbow: 4
# Left Wrist: 5
# Right Shoulder: 6
# Right Elbow: 7
# Right Wrist: 8
# Left Hip: 9
# Left Knee: 10
# Left Ankle: 11
# Right Hip: 12
# Right Knee: 13
# Right Ankle：14
train_tensor, train_label = torch.load('../dataset/train.pkl') # 导入数据集
# 原始数据集的关节点坐标都为世界坐标系，数值很大且分布分散
# 所以定义一个class，用于将世界笛卡尔坐标系转为相对center的极坐标系，其中选spine为center
class Cartesian2Polar():
    def __init__(self, dataset):
        # 如果传入的dataset是tensor，则将其转为numpy
        if dataset.type() == 'torch.DoubleTensor' or dataset.type() == 'torch.IntTensor':
            self.dataset = dataset.numpy() # tensor转numpy，便于操作
            
    def coordinate2distance(self): # 计算每个关节点相对center的欧氏距离
        normalized_train_numpy = []
        for train_data in self.dataset: # 遍历所有数据
            normalized_train_data = []
            for body_points in train_data: # 每个train_data中有32帧的关节点三维坐标信息，遍历每一帧
                center = body_points[2] # 把spine作为人体中心
                normalized_body_points = []
                distance_max = 0
                for points in body_points: # 遍历一帧中的每个关节点
                    distance = np.sqrt(np.sum(np.square(points - center))) # 求每个关节点到center的欧氏距离
                    if distance > distance_max: # 求得这一组距离中的最大值，用于后续归一化
                        distance_max = distance
                        
                    normalized_body_points.append(distance) # 把当前关节点相对center的坐标存入
                normalized_body_points /= distance_max # 归一化处理
                normalized_train_data.append(normalized_body_points) # 把15个关节点的相对归一化坐标存入（15关节点组成一帧）
            normalized_body_points = np.array(normalized_body_points)
            # normalized_train_data = np.array(normalized_train_data)
           
            normalized_train_numpy.append(normalized_train_data) # 把32帧的关节点相对归一化坐标存入（32帧组成一个视频）
        normalized_train_numpy = np.array(normalized_train_numpy)
        
        normalized_train_tensor = torch.tensor(normalized_train_numpy) # 把numpy重新转为tensor供神经网络使用
        return normalized_train_tensor
     
    def coordinate2angle(self): # 计算每个关节点相对center的两个angle
        normalized_train_numpy = [] 
        for train_data in self.dataset: # 遍历所有数据
            normalized_train_data = []
            for body_points in train_data: # 每个train_data中有32帧的关节点三维坐标信息，遍历每一帧
                center = body_points[2] # 把spine作为人体中心
                normalized_body_points = []
                for idx, points in enumerate(body_points): # 遍历一帧中的每个关节点
                    # 求得当前关节点与center再三维坐标系中的相对角度
                    if idx != 2:
                        # angle_xoy = np.arctan((points[2] - center[2])/np.sqrt(np.square(points[1] - center[1]) + np.square(points[0] - center[0])))
                        # angle_xoz = np.arctan((points[1] - center[1])/np.sqrt(np.square(points[0] - center[0]) + np.square(points[2] - center[2])))
                        # angle_yoz = np.arctan((points[0] - center[0])/np.sqrt(np.square(points[1] - center[1]) + np.square(points[2] - center[2])))   
                        angle_1 = np.arctan((points[0] - center[0]) / (points[1] - center[1]))
                        angle_2 = np.arctan((points[2] - center[2])/np.sqrt(np.square(points[0] - center[0]) + np.square(points[1] - center[1])))
                    else:
                        angle_1 = angle_2 = 0
                    normalized_body_points.append(np.array((angle_1, angle_2))) # 把当前关节点相对center的角度存入
                normalized_train_data.append(normalized_body_points) # 把15个关节点的相对归一化坐标存入（15关节点组成一帧）
            normalized_body_points = np.array(normalized_body_points)
            # normalized_train_data = np.array(normalized_train_data)
           
            normalized_train_numpy.append(normalized_train_data) # 把32帧的关节点相对归一化坐标存入（32帧组成一个视频）
        normalized_train_numpy = np.array(normalized_train_numpy)
        
        normalized_train_tensor = torch.tensor(normalized_train_numpy) # 把numpy重新转为tensor供神经网络使用
        return normalized_train_tensor


coordinate_normalization = Cartesian2Polar(train_tensor)
normalized_polar_distance = coordinate_normalization.coordinate2distance()
normalized_polar_angle = coordinate_normalization.coordinate2angle()
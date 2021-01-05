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
# train_tensor, train_label = torch.load('../dataset/train.pkl') # 导入数据集
# 原始数据集的关节点坐标都为世界坐标系，数值很大且分布分散
# 所以定义一个class，用于将世界笛卡尔坐标系转为
# 1:相对center的极坐标
# 2:相对向心节点的极坐标
# 3:相对center的笛卡尔坐标
# 4:相对向心节点的笛卡尔坐标
class Coordinate_transform():
    def __init__(self, dataset):
        # 如果传入的dataset是tensor，则将其转为numpy
        if dataset.type() == 'torch.DoubleTensor' or dataset.type() == 'torch.IntTensor':
            self.dataset = dataset.numpy() # tensor转numpy，便于操作
            
    def Center_Polar(self): # 计算每个关节点相对center的极坐标
        normalized_distance = []
        normalized_angle = [] 
        for data in self.dataset: # 遍历所有数据
            normalized_data_distance = []
            normalized_data_angle = []
            for body_points in data: # 每个train_data中有32帧的关节点三维坐标信息，遍历每一帧
                center = body_points[2] # 把spine作为人体中心
                normalized_body_points_distance = []
                normalized_body_points_angle = []
                distance_max = 0
                for idx, points in enumerate(body_points):  # 遍历一帧中的每个关节点
                    # 求节点间距离
                    distance = np.sqrt(np.sum(np.square(points - center))) # 求每个关节点到center的欧氏距离
                    if distance > distance_max: # 求得这一组距离中的最大值，用于后续归一化
                        distance_max = distance
                    
                    # 求节点间角度
                    if idx != 2: 
                        # angle_xoy = np.arctan((points[2] - center[2])/np.sqrt(np.square(points[1] - center[1]) + np.square(points[0] - center[0])))
                        # angle_xoz = np.arctan((points[1] - center[1])/np.sqrt(np.square(points[0] - center[0]) + np.square(points[2] - center[2])))
                        # angle_yoz = np.arctan((points[0] - center[0])/np.sqrt(np.square(points[1] - center[1]) + np.square(points[2] - center[2])))   
                        angle_1 = np.arctan((points[0] - center[0]) / (points[1] - center[1]))
                        angle_2 = np.arctan((points[2] - center[2])/np.sqrt(np.square(points[0] - center[0]) + np.square(points[1] - center[1])))
                    else:
                        angle_1 = angle_2 = 0
                    
        # 保存distance信息与angle信息
                    normalized_body_points_distance.append(distance) # 把当前关节点相对center的坐标存入
                    normalized_body_points_angle.append(np.array((angle_1, angle_2))) # 把当前关节点相对center的角度存入
                
                normalized_body_points_distance /= distance_max # 归一化处理
                normalized_data_distance.append(normalized_body_points_distance) # 把15个关节点组成的一帧的相对归一化坐标存入（15关节点组成一帧）
                normalized_data_angle.append(normalized_body_points_angle) # 把15个关节点的相对归一化坐标存入（15关节点组成一帧）
            
            normalized_distance.append(normalized_data_distance) # 把32帧的关节点相对归一化坐标存入（32帧组成一个视频）
            normalized_angle.append(normalized_data_angle) # 把32帧的关节点相对角度存入（32帧组成一个视频）
        
        normalized_distance = np.array(normalized_distance)
        normalized_angle = np.array(normalized_angle)
        
        normalized_distance_tensor = torch.tensor(normalized_distance) # 把numpy重新转为tensor供神经网络使用
        normalized_angle_tensor = torch.tensor(normalized_angle) # 把numpy重新转为tensor供神经网络使用
        
        return normalized_distance_tensor, normalized_angle_tensor

    
    def Relative_Polar(self): # 转换为相对向心节点的极坐标
        dic = {0:1, 3:1, 6:1, 4:3, 5:4, 7:6, 8:7, 2:2, 1:2, 9:2, 12:2, 10:9, 11:10, 13:12, 14:13}
        normalized_angle = [] 
        normalized_distance = [] 
        for train_data in self.dataset: # 遍历所有172个视频数据
            normalized_data_distance = []
            normalized_data_angle = []
            for idx, body_points in enumerate(train_data): # 每个视频有32帧，遍历
                distance_max = 0
                normalized_body_points_distance = []
                normalized_body_points_angle = []
                for idx, points in enumerate(body_points): # 遍历一帧中的每个关节点
                    center = body_points[dic[idx]] # 把spine作为人体中心
                    # 求节点间距离
                    distance = np.sqrt(np.sum(np.square(points - center))) # 求每个关节点到center的欧氏距离
                    if distance > distance_max: # 求得这一组距离中的最大值，用于后续归一化
                        distance_max = distance
                    # 求得当前关节点与center再三维坐标系中的相对角度
                    if idx != 2:
                        # angle_xoy = np.arctan((points[2] - center[2])/np.sqrt(np.square(points[1] - center[1]) + np.square(points[0] - center[0])))
                        # angle_xoz = np.arctan((points[1] - center[1])/np.sqrt(np.square(points[0] - center[0]) + np.square(points[2] - center[2])))
                        # angle_yoz = np.arctan((points[0] - center[0])/np.sqrt(np.square(points[1] - center[1]) + np.square(points[2] - center[2])))   
                        angle_1 = np.arctan((points[0] - center[0]) / (points[1] - center[1]))
                        angle_2 = np.arctan((points[2] - center[2])/np.sqrt(np.square(points[0] - center[0]) + np.square(points[1] - center[1])))
                    else:
                        angle_1 = angle_2 = 0
                        
                    normalized_body_points_distance.append(distance) # 把当前关节点相对center的坐标存入
                    normalized_body_points_angle.append(np.array((angle_1, angle_2))) # 把当前关节点相对center的角度存入
                
                normalized_body_points_distance /= distance_max # 归一化处理
                normalized_data_distance.append(normalized_body_points_distance) # 把15个关节点组成的一帧的相对归一化坐标存入（15关节点组成一帧）
                normalized_data_angle.append(normalized_body_points_angle) # 把15个关节点的相对归一化坐标存入（15关节点组成一帧）
           
            normalized_distance.append(normalized_data_distance) # 把32帧的关节点相对归一化坐标存入（32帧组成一个视频）
            normalized_angle.append(normalized_data_angle) # 把32帧的关节点相对角度存入（32帧组成一个视频）
        
        normalized_distance = np.array(normalized_distance)
        normalized_angle = np.array(normalized_angle)
        
        normalized_distance_tensor = torch.tensor(normalized_distance) # 把numpy重新转为tensor供神经网络使用
        normalized_angle_tensor = torch.tensor(normalized_angle) # 把numpy重新转为tensor供神经网络使用
        return normalized_distance_tensor, normalized_angle_tensor

    
    def Relative_Cartesian(self): # 转换为相对向心节点的笛卡尔坐标
        dic = {0:1, 3:1, 6:1, 4:3, 5:4, 7:6, 8:7, 2:2, 1:2, 9:2, 12:2, 10:9, 11:10, 13:12, 14:13}
        normalized_location = [] 
        for train_data in self.dataset: # 遍历所有172个视频数据
            normalized_data_location = []
            for idx, body_points in enumerate(train_data): # 每个视频有32帧，遍历
                distance_max = 0
                normalized_body_points_location = []
                for idx, points in enumerate(body_points): # 遍历一帧中的每个关节点
                    center = body_points[dic[idx]] # 根据预先建立的字典，查找与当前节点对应的向心节点
                    # 求节点相对坐标
                    location = points - center
                    distance = np.sqrt(np.sum(np.square(location))) # 求每个关节点到center的欧氏距离
                    if distance > distance_max: # 求得这一组距离中的最大值，用于后续归一化
                        distance_max = distance
                        
                    normalized_body_points_location.append(location) # 把当前关节点相对center的坐标存入
                
                normalized_body_points_location /= distance_max # 归一化处理
                normalized_data_location.append(normalized_body_points_location) # 把15个关节点组成的一帧的相对归一化坐标存入（15关节点组成一帧）

           
            normalized_location.append(normalized_data_location) # 把32帧的关节点相对归一化坐标存入（32帧组成一个视频）
        
        normalized_location = np.array(normalized_location)
        
        normalized_location_tensor = torch.tensor(normalized_location) # 把numpy重新转为tensor供神经网络使用

        return normalized_location_tensor
    
    
    def Center_Cartesian(self): # 转换为相对center的笛卡尔坐标
        normalized_location = [] 
        for train_data in self.dataset: # 遍历所有172个视频数据
            normalized_data_location = []
            for idx, body_points in enumerate(train_data): # 每个视频有32帧，遍历
                center = body_points[2] # 把spine作为人体中心
                distance_max = 0
                normalized_body_points_location = []
                for idx, points in enumerate(body_points): # 遍历一帧中的每个关节点
                    
                    # 求节点相对坐标
                    location = points - center
                    distance = np.sqrt(np.sum(np.square(location))) # 求每个关节点到center的欧氏距离
                    if distance > distance_max: # 求得这一组距离中的最大值，用于后续归一化
                        distance_max = distance
                        
                    normalized_body_points_location.append(location) # 把当前关节点相对center的坐标存入
                
                normalized_body_points_location /= distance_max # 归一化处理
                normalized_data_location.append(normalized_body_points_location) # 把15个关节点组成的一帧的相对归一化坐标存入（15关节点组成一帧）

           
            normalized_location.append(normalized_data_location) # 把32帧的关节点相对归一化坐标存入（32帧组成一个视频）
        
        normalized_location = np.array(normalized_location)
        
        normalized_location_tensor = torch.tensor(normalized_location) # 把numpy重新转为tensor供神经网络使用

        return normalized_location_tensor

# coordinate_normalization = Cartesian2Polar(train_tensor)
# normalized_polar_distance = coordinate_normalization.coordinate2distance()
# normalized_polar_angle = coordinate_normalization.coordinate2angle()
# relative_polar = coordinate_normalization.relative_polar()
# Relative_Cartesian = coordinate_normalization.Relative_Cartesian()
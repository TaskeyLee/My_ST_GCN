import torch
import numpy as np
import random

# 每行48个元素，4~48为15个人体关键点的三维坐标，1~3为视频序号
# 1：视频id，共215
# 2：视频参与者id，共10
# 3：动作类别，共10
# 表示为：不同的动作类别，分别让10个人都做2/3遍，做了几遍就是视频id
# 数据集中0,1,2项都一致时，则表示的是同一段视频中不同帧的情况
file_name = '../dataset/Florence_3d_actions/Florence_dataset_WorldCoordinates.txt'
f = open(file_name)
lines = f.readlines()
prev_video = int(lines[0][0])
prev_categ = int(lines[0][2])
frames = []
train = []
valid = []
test  = []
train_label = []
valid_label = []
test_label  = []
for line in lines:
	line = line.split(' ')
	vid = int(line[0])
	aid = int(line[1])
	cid = int(line[2])-1

	features = list(map(float, line[3:])) 
	#norm_val = float(line[-1])

	if prev_video == vid: # 当满足
		frames.append(np.reshape(np.asarray(features), (-1,3)))
	else:
		if len(frames) >= 32:
			frames = random.sample(frames, 32)
			frames = torch.from_numpy(np.stack(frames, 0))
		else:
			frames = np.stack(frames, 0)
			xloc = np.arange(frames.shape[0])
			new_xloc = np.linspace(0, frames.shape[0], 32)
			frames = np.reshape(frames, (frames.shape[0], -1)).transpose()

			new_datas = []
			for data in frames:
				new_datas.append(np.interp(new_xloc, xloc, data))
			frames = torch.from_numpy(np.stack(new_datas, 0)).t()

		frames = frames.view(32, -1, 3)
		if prev_actor < 9: # 参与者标签小于9的分为训练集
			train.append(frames)
			train_label.append(prev_categ)
			#train.append(torch.stack([frames[:,:,2],frames[:,:,1],frames[:,:,0]], 2))
			#train_label.append(prev_categ)
			#train.append(torch.stack([frames[:,:,0],frames[:,:,2],frames[:,:,1]], 2))
			#train_label.append(prev_categ)
			#train.append(torch.cat([frames[:,:3,:],frames[:,6:9,:],frames[:,3:6,:],
			#						frames[:,12:15,:],frames[:,9:12,:]], 1))
			#train_label.append(prev_categ)
		elif prev_actor < 10: # 参与者标签为9的分为验证集
			valid.append(frames)
			valid_label.append(prev_categ)
		else: # 参与者标签为10的分为测试集
			test.append(frames)
			test_label.append(prev_categ)
		frames = [np.reshape(np.asarray(features), (-1,3))]
	prev_actor = aid # 录这视频的参与者id
	prev_video = vid # 视频id
	prev_categ = cid # 动作类别id
	

if len(frames) >= 32:  # 每个视频被统一分割为32帧，当32帧满后，进行一些处理
	frames = random.sample(frames, 32)
	frames = torch.from_numpy(np.stack(frames, 0))
else:
	frames = np.stack(frames, 0)
	xloc = np.arange(frames.shape[0])
	new_xloc = np.linspace(0, frames.shape[0], 32)
	frames = np.reshape(frames, (frames.shape[0], -1)).transpose()

	new_datas = []
	for data in frames:
		new_datas.append(np.interp(new_xloc, xloc, data))
	frames = torch.from_numpy(np.stack(new_datas, 0)).t()

	
frames = frames.view(32, -1, 3)
if aid < 9:
	train.append(frames)
	train_label.append(prev_categ)
	#train.append(torch.stack([frames[:,:,2],frames[:,:,1],frames[:,:,0]], 2))
	#train_label.append(prev_categ)
	#train.append(torch.stack([frames[:,:,0],frames[:,:,2],frames[:,:,1]], 2))
	#train_label.append(prev_categ)
	#train_label.append(prev_categ)
	#train.append(torch.cat([frames[:,:3,:],frames[:,6:9,:],frames[:,3:6,:],
	#						frames[:,12:15,:],frames[:,9:12,:]], 1))
	#train_label.append(prev_categ)
elif aid < 10:
	valid.append(frames)
	valid_label.append(prev_categ)
else:
	test.append(frames)
	test_label.append(prev_categ)

train_label = torch.from_numpy(np.asarray(train_label))
valid_label = torch.from_numpy(np.asarray(valid_label))
test_label  = torch.from_numpy(np.asarray(test_label))

torch.save((torch.stack(train, 0), train_label), './dataset/train.pkl')
torch.save((torch.stack(valid, 0), valid_label), './dataset/valid.pkl')
torch.save((torch.stack(test, 0),  test_label),  './dataset/test.pkl')
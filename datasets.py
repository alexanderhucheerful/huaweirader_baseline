#author :alexhu
#time: 2020.4.30
#欢迎各位大佬指点交流：qq-》2473992731
import os
import random
import numpy as np
import PIL.Image as Image
import glob
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
import time
#from  normlize_train_test.filter_dbz import filtrate_dbz

class 	TrainloadedDataset(Dataset):
	def __init__(self, root_dir,pd_path,npy_path,frame_num=None, time_freq = None, ps = None):
		#the pd path is your pkl location  root_dir is your png loaction
		#这两个参数是用来控制滑窗变量，因为一个样本只能切出来一个seq所以不使用滑窗
		self.fnum = frame_num
		self.freq = time_freq
		self.root_dir = root_dir
		self.npy_path = npy_path
		self.ps = ps
		self._df = pd.read_pickle(pd_path)
		npydata = np.load(self.npy_path)
		npydata = list(npydata)
		self.index = npydata


	def __len__(self):

		return  len(self.index)

	def __getitem__(self, idx):
		tid = self.index[idx]
		#input = (1-19),target(25,30,35,40)
		ids1 = list(range(tid + 1, tid + 20 ,2))
		ids2 = list(range(tid + 25,tid + 45 ,5))
		ids = ids1 + ids2

		inputframes = []
		targetframes = []
		for ctid in ids1:
			try:
				frame = np.array(Image.open(os.path.join(self.root_dir, self._df['fname'][ctid])))
			except:
				frame = np.array(Image.open(os.path.join(self.root_dir, self._df['fname'][ctid-1])))
			frame = np.expand_dims(frame,axis = 0)
			#是否有其他的fill_nan方式
			frame[frame>=80.5] = 0
			frame = frame/80.0


			#frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])
			
			#frame = frame[:,0:480,0:480]
			#print(frame.shape)
			inputframes.append(frame)
			#frames_crop = frames

		for ctid in ids2:
			try:
				frame = np.array(Image.open(os.path.join(self.root_dir, self._df['fname'][ctid])))
			except:
				frame = np.array(Image.open(os.path.join(self.root_dir, self._df['fname'][ctid-1])))
			frame = np.expand_dims(frame,axis = 0)
			#是否有其他的fill_nan方式
			frame[frame>=80.5] = 0
			frame = frame/80.0


			#frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])
			
			#frame = frame[:,0:480,0:480]
			#print(frame.shape)
			targetframes.append(frame)

		inputframes = torch.from_numpy(np.array(inputframes ))
		targetframes = torch.from_numpy(np.array(targetframes ))
		return ids, (inputframes, targetframes)

	def _get_patch(self, imgs):
		H = imgs[0].shape[1]
		W = imgs[0].shape[2]

		if self.ps < W and self.ps < H:
			xx = np.random.randint(0, W-self.ps)
			yy = np.random.randint(0, H-self.ps)
		
			imgs_crop = []
			for img in imgs:
				img_crop = img[:, yy:yy+self.ps, xx:xx+self.ps]
				imgs_crop.append(img_crop)
		else:
			imgs_crop = imgs

		if np.random.randint(2, size=1)[0] == 1:
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.flip(imgs_crop[i], axis=2).copy()
		if np.random.randint(2, size=1)[0] == 1: 
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.flip(imgs_crop[i], axis=1).copy()
		if np.random.randint(2, size=1)[0] == 1:
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.transpose(imgs_crop[i], (0, 2, 1)).copy()
		
		return imgs_crop



class 	TestloadedDataset(Dataset):
	def __init__(self, root_dir,pd_path,npy_path,frame_num=None, time_freq = None, ps = None):
		#the pd path is your pkl location  root_dir is your png loaction
		#这两个参数是用来控制滑窗变量，因为一个样本只能切出来一个seq所以不使用滑窗
		self.fnum = frame_num
		self.freq = time_freq
		self.root_dir = root_dir
		self.npy_path = npy_path
		self.ps = ps
		self._df = pd.read_pickle(pd_path)
		npydata = np.load(self.npy_path)
		npydata = list(npydata)
		self.index = npydata


	def __len__(self):

		return  len(self.index)

	def __getitem__(self, idx):
		tid = self.index[idx]
		#input = (1-19),target(25,30,35,40)
		ids1 = list(range(tid + 1, tid + 20 ,2))
		#ids2 = list(range(tid + 25,tid + 45 ,5))
		#ids = ids1 + ids2
		test_df = self._df
		test_index = list(map(lambda x:x[0].split('/')[-2],self._df.values))
		test_df.index = [test_index,test_index]
		test_df.index.names=['index0','index1']
		filename = test_df['fname'].keys()[tid+1][0]

		inputframes = []
		targetframes = []
		for ctid in ids1:
			try:
				frame = np.array(Image.open(os.path.join(self.root_dir, self._df['fname'][ctid])))
			except:
				frame = np.array(Image.open(os.path.join(self.root_dir, self._df['fname'][ctid-1])))
			frame = np.expand_dims(frame,axis = 0)
			#是否有其他的fill_nan方式
			frame[frame>=80.5] = 0
			frame = frame/80.0


			#frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])
			
			#frame = frame[:,0:480,0:480]
			#print(frame.shape)
			inputframes.append(frame)
			#frames_crop = frames

		inputframes = torch.from_numpy(np.array(inputframes ))

		return filename, inputframes

	def _get_patch(self, imgs):
		H = imgs[0].shape[1]
		W = imgs[0].shape[2]

		if self.ps < W and self.ps < H:
			xx = np.random.randint(0, W-self.ps)
			yy = np.random.randint(0, H-self.ps)
		
			imgs_crop = []
			for img in imgs:
				img_crop = img[:, yy:yy+self.ps, xx:xx+self.ps]
				imgs_crop.append(img_crop)
		else:
			imgs_crop = imgs

		if np.random.randint(2, size=1)[0] == 1:
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.flip(imgs_crop[i], axis=2).copy()
		if np.random.randint(2, size=1)[0] == 1: 
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.flip(imgs_crop[i], axis=1).copy()
		if np.random.randint(2, size=1)[0] == 1:
			for i in range(len(imgs_crop)):
				imgs_crop[i] = np.transpose(imgs_crop[i], (0, 2, 1)).copy()
		
		return imgs_crop

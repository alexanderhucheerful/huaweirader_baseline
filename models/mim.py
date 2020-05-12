

import torch
import torch.nn as nn
import sys
sys.path.append("/media/workdir/hujh/hujh-new/huaweirader_baseline/models")
from layers.SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as stlstm
from layers.MIMBlock import MIMBlock as mimblock
from layers.MIMN import MIMN as mimn
import math


class MIM(nn.Module): # stlstm 
	def __init__(self, shape, num_layers, num_hidden, filter_size,
		total_length=15, input_length=10, tln=True):
		super(MIM, self).__init__()
		
		self.num_layers = num_layers
		self.num_hidden = num_hidden
		self.filter_size = filter_size
		self.total_length = total_length
		self.input_length = input_length
		self.tln = tln
		

		self.stlstm_layer = nn.ModuleList() # 存储 stlstm 和 mimblock
		self.stlstm_layer_diff = nn.ModuleList() # 存储 mimn

		self.shape = shape # 输入形状
		self.output_channels = shape[-3] # 输出的通道数
		
		for i in range(self.num_layers): # 隐藏层数目
			if i == 0:
				num_hidden_in = self.num_hidden[self.num_layers - 1] # 隐藏层的输入 前一时间段最后一层的输出为后一时间段第一层的输入
			else:
				num_hidden_in = self.num_hidden[i - 1] # 隐藏层的输入
			if i < 1: # 初始层 使用 stlstm
				new_stlstm_layer = stlstm('stlstm_' + str(i + 1),
							  self.filter_size,
							  num_hidden_in,
							  self.num_hidden[i],
							  self.shape,
							  self.output_channels,
							  tln=self.tln)
			else: # 后续层 使用 mimblock
				new_stlstm_layer = mimblock('stlstm_' + str(i + 1),
								self.filter_size,
								num_hidden_in,
								self.num_hidden[i],
								self.shape,
								self.num_hidden[i-1],
								tln=self.tln)
			self.stlstm_layer.append(new_stlstm_layer) # 列表


		for i in range(self.num_layers - 1): # 添加 MIMN
			new_stlstm_layer = mimn('stlstm_diff' + str(i + 1),
								self.filter_size,
								self.num_hidden[i + 1],
								self.shape,
								tln=self.tln)
			self.stlstm_layer_diff.append(new_stlstm_layer) # 列表



		
		# 生成图片
		self.x_gen = nn.Conv2d(self.num_hidden[self.num_layers - 1],
				 self.output_channels,1,1,padding=0
				 )
		
		#下采样
		self.downsample = nn.Sequential(nn.Conv2d(1,1,5,4,1),
										nn.LeakyReLU(negative_slope=0.2, inplace=True))

		#上采样
		self.upsample = nn.Sequential(nn.ConvTranspose2d(1,1,6,4,1),
									  nn.LeakyReLU(negative_slope=0.2, inplace=True),
									  nn.Conv2d(1,8,3,1,1),
									  nn.LeakyReLU(negative_slope=0.2, inplace=True),
									  nn.Conv2d(8,1,1,1,0))
	def forward(self, images):
		 # 存储生成的图片
		batch_size, seq_number,input_channel, height, width = images.size()
        #images = torch.reshape(images, (-1, input_channel, height, width))
		images = torch.reshape(images, (-1, input_channel, height, width))
		images = self.downsample(images)
		images = torch.reshape(images, (batch_size, seq_number, images.size(1),
                                        images.size(2), images.size(3)))
		print(images.shape)
		st_memory = None
		St=[]#存储S
		gen_images=[]
		cell_state = []  # 存储 stlstm_layer 的记忆
		hidden_state = []  # 存储 stlstm_layer 的隐藏层输出
		cell_state_diff = []  # 存储 stlstm_layer_diff 的记忆
		hidden_state_diff = []  # 存储 stlstm_layer_diff 的隐藏层输出
		for i in range(self.num_layers):
			cell_state.append(None)  # 记忆
			hidden_state.append(None)  # 状态
			St.append(None)
		for i in range(self.num_layers-1):
			
			cell_state_diff.append(None)  # 记忆
			hidden_state_diff.append(None)  # 状态
		for time_step in range(self.total_length - 1): # 时间步长

			
			if time_step < self.input_length: # 小于输入步长
				x_gen = images[:,time_step] # 输入大小为 [batch, in_channel,in_height, in_width]
			else:
				# 掩模 mask
				# print(schedual_sampling_bool)
				# x_gen = schedual_sampling_bool[:,time_step-self.input_length]*images[:,time_step] + \
				# 		(1-schedual_sampling_bool[:,time_step-self.input_length])*x_gen
				x_gen=x_gen

						
			preh = hidden_state[0] # 初始化状态
			hidden_state[0],cell_state[0],st_memory = self.stlstm_layer[0]( # 使用的是 stlstm 输出 hidden_state[0], cell_state[0], st_memory
				x_gen, hidden_state[0], cell_state[0], st_memory)
			
			# 对于 mimblock
			for i in range(1, self.num_layers):

				if time_step > 0:
					if i == 1:
						hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1]( # 先求出 mimn
							hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
					else:
						hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1]( # 先求出 mimn
							hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
				else:
					self.stlstm_layer_diff[i - 1](torch.zeros_like(hidden_state[i - 1]), None, None)
				
				# 接下来计算 mimblock	
				preh = hidden_state[i]
				hidden_state[i], cell_state[i], st_memory,St[i] = self.stlstm_layer[i]( # mimblock
					hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i], st_memory,St[i])
				
			# 生成图像 取最后一层的隐藏层状态
			x_gen = self.x_gen(hidden_state[self.num_layers - 1])
			x_gen_out = self.upsample(x_gen)
			
			gen_images.append(x_gen_out)

		gen_image = torch.stack(gen_images, dim=1)
		gen_image = gen_image[:,10:]

		
		return (gen_image,)

		
if __name__ == '__main__':
	x = torch.rand(2,10,1,256,256).cuda()
	target = torch.rand(2,4,1,256,256).cuda()
	model = MIM((2,10,1,64,64),4,(64,64,64,64),5).cuda()
	#y = model(x)
	#y = y[:,10:]
	#print(y.shape)
	#print(y.shape)
	torch.cuda.empty_cache()
	lossfunc = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	for epoch in range(1000):
		out =  model(x)
		out = out[:,10:]
		loss = lossfunc(out,target)
		optimizer.zero_grad()  
		loss.backward()   
		optimizer.step()   
		print(epoch, 'loss:', loss.item())
		torch.cuda.empty_cache()
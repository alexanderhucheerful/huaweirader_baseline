#author :alexhu
#time: 2020.4.30
#欢迎各位大佬指点交流：qq-》2473992731


import sys
sys.path.insert(0, '..')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import time
import pickle
import copy
from datasets import TestloadedDataset
import pandas as pd
#from models.modelconv3d import Network
import os
import glob
import argparse
import matplotlib
matplotlib.use("Pdf")
from matplotlib import colors
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from utils import AverageMeter, save_checkpoint, adjust_learning_rate
#from losses import RMSE_Loss
import ipdb
import os
from models.PytorchUNet.unet.unet_model import UNet
import scipy.misc
import PIL.Image as Image

#####################
"""
文档说明
同train.py
测试时顺序加载数据，batchsize为1，不shuffle
请同样严格遵循数据接口
需要改路径或名称的地方将插入changepoint，顺序搜索即可

"""
#####################
parser = argparse.ArgumentParser(description = 'Test')
parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
args = parser.parse_args()

##### the modelparameter path is :
#changepoint
unetsave_dir = '/media/workdir/hujh/hujh-new/huaweirader_baseline/model_parameters/classify_test/unet_model/'
test_npy_path  = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/test_demo.npy'
#convgrusave_dir = '/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/model_saving/convgru_bestmodel/'
valid_result_dir ='/media/workdir/hujh/hujh-new/huaweitest/'
valid_dir = ''
valid_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/test_all_pkl.pkl'



valid_datasets = []
valid_loaders = []
all_datasets = TestloadedDataset(valid_dir, valid_path,test_npy_path  )
valid_loaders =  torch.utils.data.DataLoader(all_datasets,batch_size  =1,shuffle=False)
#v_sample = torch.utils.data.SubsetRandomSampler(indices)
#valid_loaders =  torch.utils.data.DataLoader(valid_datasets,batch_size  =args.batch_size,sampler=v_sample)
print("the Testing loader is finish")


######  load the model to compare###############################################################
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')
device3 = torch.device('cuda:3')
device = ['cuda:1','cuda:3','cuda:2','cuda:0']


model_unet = UNet(10,4).to(device1)

############## 请以如下方式加载集成模型################
#changepoint
if os.path.exists(os.path.join(unetsave_dir, 'unetcheckpoint.pth.tar')):
    # load existing model
    print('==> loading existing model')
    g_info = torch.load(os.path.join(unetsave_dir, 'unetcheckpoint.pth.tar'))
    model_unet.load_state_dict(g_info['state_dict'])
else:
    print('==> No test model detected!')
    exit(1)

"""
convgru_encoder = encoder(convgru_encoder_params[0], convgru_encoder_params[1]).to(device3)
convgru_decoder = decoder(convgru_forecaster_params[0], convgru_forecaster_params[1]).to(device3)
model_convgru = EF(convgru_encoder, convgru_decoder).to(device3)
if os.path.exists(os.path.join(convgrusave_dir, 'convgrucheckpoint.pth.tar')):
    # load existing model
    print('==> loading existing model')
    g_info = torch.load(os.path.join(convgrusave_dir, 'convgrucheckpoint.pth.tar'))
    model_convgru.load_state_dict(g_info['state_dict'])
else:
    print('==> No test model detected!')
    exit(1)
"""

#####################################请严格以如下有序字典方式添加集合成员###################################################
models = OrderedDict({
    'unet': model_unet

})


######### mian #############
#model_run_avarage_time = dict()
with torch.no_grad():
    current = 0
    for name,model in models.items():
        is_deeplearning_model = (torch.nn.Module in model.__class__.__bases__)
        if is_deeplearning_model:
            model.eval()
        
        
        for ind, (filename, inputframes) in enumerate(valid_loaders):
            print(filename)
            ind =ind+1
            savepath = os.path.join(valid_result_dir,filename[0])
            if not os.path.isdir(savepath):
                os.makedirs(savepath)

            #####################  请严格按照集合模型的输入格式进行转换-bscwh ###########
                                ##### 数据入口#######
            if name =='unet':
                test_inputs = inputframes.squeeze(2)
            ############################################################################

            test_inputs = test_inputs.type(torch.FloatTensor).to(device1)
            predict = model_unet(test_inputs)

            ######################  数据出口bscwh即(1,4,256,256) ############
            if name =='unet':
                predict = predict.unsqueeze(2)
            #################################################################
            predict_np = np.clip(predict.detach().cpu().numpy(), 0, 1)
            predict_np  = predict_np * 80.0
            predict_np = np.squeeze(predict_np)
            #predict_np[predict_np<0.01] = 255
            timefilename = [30,60,90,120]
            #print(predict_np.shape)
            for time in range(predict_np.shape[0]):
                #print(time)
                picnumber =timefilename[time]
                picname = str(picnumber)+'.png'
                pic_savepath = os.path.join(savepath,picname)
                #scipy.misc.toimage(predict_np, high=255, low=0, cmin=0, cmax=255).save(pic_savepath)
                predict_np_save = predict_np[time,:,:]
                #print(predict_np_save.shape)
                Image.fromarray(np.uint8(predict_np_save)).save(pic_savepath)
print('passing unity testing')



            




            


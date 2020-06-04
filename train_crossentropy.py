#author :alexhu
#time: 2020.4.30
#欢迎各位大佬指点交流：qq-》2473992731

from __future__ import division
from __future__ import print_function
import os, time, scipy, shutil, sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import glob
import argparse
import logging
import pandas as pd
from utils import AverageMeter, save_checkpoint, adjust_learning_rate
from datasets import TrainloadedDataset
#from losses import RMSE_Loss
from models.PytorchUNet.unet.unet_model import UNet
from models.predrnn import predrnned
from zhanghangparallel import DataParallelModel,DataParallelCriterion
import matplotlib
matplotlib.use("Pdf")
from matplotlib import colors
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
#from loss.loss import Weighted_mse_mae
#from trajgru.params import convgru_encoder_params,convgru_forecaster_params
#from trajgru.encoder import encoder
##from trajgru.decoder import decoder
#from trajgru.model import EF
#from trajgru.radam import RAdam
#from ranger import Ranger
from torch.optim import lr_scheduler
import ipdb
import pickle
from alexhuall_lossfuncation import Bmsemae 
from pytorchtools import EarlyStopping
from ranger import Ranger
#from alexhuall_lossfuncation import gdl,SSIM
from alexhuall_lossfuncation import draw_video
from alexhuall_lossfuncation import RMSE_Loss
from alexhuall_lossfuncation import mixloass ,SoftIoULoss,MulticlassDiceLoss
from models.crossmodel.unet_model import UNet

#############文档说明############
#标准数据结构从datasets返回为 batchsize*seq*channel*width*height,请十分注意，且精度是doubletensor需要转化为floattensor
#直觉告诉我unet会在这个任务里效果显著
#可以尝试的方向有 predrnn++，mim，e3d，selfatten_convgru等集合成员
#本次训练采用early_stoping策略，ranger优化器，lr2个epoch衰减0.7
#训练测试集比为8：2随机划分
"""
本demo模板需要改动的路径如下:
1.各种储存路径
2.模型参数checkpoint的名称
3.数据接口，请务必高度封装你的模型，如encoder-decoer写成一个class不要散着写，可以直接导入train.py中
我将在每个需要改动的的地方插入changepoint断点请在ide里直接顺序find：changepoint字符串
"""
###################################################
parser = argparse.ArgumentParser(description = 'Train')
#################  benachmark setting##############
# training parameters
parser.add_argument('--batch_size', default=12, type=int, help='mini-batch size')
parser.add_argument('--patch_size', default=256, type=int, help='image patch size')
parser.add_argument('-lr', default=1e-3, type=float, help='G learning rate')
parser.add_argument('--grad_clip', type=float, default=50, help='gradient clipping')
#parser.add_argument('-frame_num', default=20, type=int, help='sum of frames')
#parser.add_argument('-time_freq', default=1, type=int, help='predict freq')
parser.add_argument('-epochs', default=1, type=int, help='sum of epochs')
#LR_step_size = 20000
# visualization setting
#parser.add_argument('-save_freq', default=10, type=int, help='save freq of visualization')
# visualizeation valid process
parser.add_argument('-valid_freq', default=1, type=int, help='save valid of visualization')
args = parser.parse_args()
# parallel model caculate poin main device for main model load
#####################################  dir set################################################
#changepoint
device = torch.device('cuda:0')
train_dir = ''
valid_dir = ''
pd_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl'
train_npy_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_demo.npy'
save_dir = '/media/workdir/hujh/hujh-new/huaweirader_baseline/model_parameters/classify_test/predrnn_model/'
#bestmodel_dir = '/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/model_saving/convgru_bestmodel/'
#result_dir = '/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/stacking_result/'
valid_result_dir = '/media/workdir/hujh/hujh-new/huaweirader_baseline/validation_video/unet_val_result/'
valid_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl'
log_dir = '/media/workdir/hujh/hujh-new/huaweirader_baseline/log/demolog'

#####################################################################################
##### set the rader echo color bar####
#changepoint
colorbar_dir = '/media/workdir/hujh/hujh-new/huaweirader_baseline/colorbar.txt'
rgb=np.loadtxt(colorbar_dir,delimiter=',')
rgb/=255.0
icmap=colors.ListedColormap(rgb,name='my_color')
cmap_color=icmap
#####################################################################################

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.isdir(valid_result_dir):
    os.makedirs(valid_result_dir)
fh = logging.FileHandler(os.path.join(valid_result_dir, 'valid.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

train_datasets = [] 
train_loaders = [] 
valid_datasets = []
valid_loaders = []

all_datasets = []


################################ load datasets ###############################
#按照切割数据集的办法来构建数据集
all_datasets = TrainloadedDataset(train_dir, pd_path,train_npy_path)
print("the train loader is ok")
random_seed = 1998
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)

train_size = int(0.8 * len(all_datasets))
valid_size = int(0.2* len(all_datasets))
#test_size = len(all_datasets)-train_size-valid_size 
train_datasets, valid_datasets = torch.utils.data.random_split(all_datasets, [train_size,valid_size])
##################train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, pin_memory=True)

train_loaders=  torch.utils.data.DataLoader(train_datasets,batch_size  =args.batch_size,shuffle=True,num_workers = 4,pin_memory=True,drop_last=True)

#print('checkpoint of len train_laoders')

valid_loaders =  torch.utils.data.DataLoader(valid_datasets,batch_size  = 4,shuffle=False,num_workers = 4,pin_memory=True,drop_last=True)

##############################################################################
######################choose your model#######################################
                           


#model = UNet(10,4)
model = predrnned()
#print(model)
#print(model.decoder.lastoutput)
########################## convert the reg to a classify quesion ###################################3#####################
class WeightedCrossEntropyLoss(nn.Module):

    # weight should be a 1D Tensor assigning weight to each of the classes.

    def __init__(self, thresholds, weight=None, LAMBDA=None):

        super().__init__()

        # 每个类别的权重，使用原文章的权重。

        self._weight = weight

        # 每一帧 Loss 递进参数

        self._lambda = LAMBDA

        # thresholds: 雷达反射率

        self._thresholds = thresholds

    # input: output prob, b*s*C*H*W

    # target: b*s*1*H*W, original data, range [0, 1]

    # mask: S*B*1*H*W

    def forward(self, input, target):

        #assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN

        # F.cross_entropy should be B*C*S*H*W

        input = input.permute((0, 2, 1, 3, 4))

        # B*S*H*W

        target = target.squeeze(2)

        class_index = torch.zeros_like(target).long()

        thresholds = self._thresholds
        self._weight = self._weight.to(target.device)

        # print(thresholds)
        class_index[...] =0
        #print(class_index.shape)
        #print(class_index)
        for i, threshold in enumerate(thresholds):
            i = i+1
            class_index[target >= threshold] = i
        #class_index = class_index
        #print(class_index)
        #print((class_index==4).all())
        print(input)
        result = torch.argmax(input, dim=1,keepdim = True)
        #print(result)

 
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')

        if self._lambda is not None:

            B, S, H, W = error.size()



            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)

            if torch.cuda.is_available():

                w = w.to(target.device)

                # B, H, W, S

            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # S*B*1*H*W

        error = error.permute(0, 1, 2, 3).unsqueeze(2)

        return torch.mean(error.float())



class multiclassdice(nn.Module):
    def __init__(self,thresholds,weight = None):
        super().__init__()
        self._weight = weight
        self.lossfunc = MulticlassDiceLoss()
        self._thresholds = thresholds
    

    def forward(self,input,target):
        input = input.permute((0, 2, 1, 3, 4))

        # B*S*H*W

        target = target.permute((0,2,1,3,4))

        class_index = torch.zeros_like(target).long()

        thresholds = self._thresholds
        self._weight = self._weight.to(target.device)

        # print(thresholds)
        class_index[...] =0
        #print(class_index.shape)
        #print(class_index)
        for i, threshold in enumerate(thresholds):
            i = i+1
            class_index[target >= threshold] = i
        onehot_classindex = self.make_one_hot(class_index,5).to(target.device)
        loss = self.lossfunc(input,onehot_classindex,self._weight)

        return loss

        

    
    def make_one_hot(self,input, num_classes):

        """Convert class index tensor to one hot encoding tensor.

        Args:

            input: A tensor of shape [N, 1, *]

            num_classes: An int of number of class

        Returns:

            A tensor of shape [N, num_classes, *]

        """

        shape = np.array(input.shape)

        shape[1] = num_classes

        shape = tuple(shape)

        result = torch.zeros(shape)

        result = result.scatter_(1, input.cpu(), 1)



        return result


class ProbToPixel(object):
    #转换类别为预测值


    def __init__(self, middle_value):



        self._middle_value = middle_value



    def __call__(self, prediction):

        '''

        prediction: 输入的类别预测值，S*B*C*H*W

        ground_truth: 实际值，像素/255.0, [0, 1]

        lr: 学习率

        :param prediction:

        :return:

        '''

        # 分类结果，0 到 classes - 1

        # prediction: b*s*C*H*W
        #self._middle_value = self._middle_value.to(prediction.device)
        print(prediction)

        result = torch.argmax(prediction, axis=2,keepdim = True)
        print(result)

        prediction_result = torch.ones_like(result,dtype = torch.float32)

        for i in range(len(self._middle_value)):

            prediction_result[result==i] = self._middle_value[i]

            # 如果需要更新替代值

            # 更新替代值
        #print(prediction_result.dtype)
        return prediction_result

"""
for param in model.parameters():

    param.requires_grad = Falseparamer
"""
model.decoder.lastoutput = nn.Sequential(nn.ConvTranspose2d(16,8,6,4,1),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv2d(8,4,3,1,1),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Conv2d(4,1,1,1,0),
                                        UNet(1,5))


thresholds = [0.25,0.375,0.5,0.625]
weights  = torch.FloatTensor([2, 3, 6, 10, 30])
midlevalue = [0.1875,0.3125,0.4375,0.5625,0.8125]
#weigths = weigths.to(device)

##################################################################################################################################3
if torch.cuda.device_count()>1:
	print("use:",torch.cuda.device_count(),"gpus")
#model = nn.DataParallel(model,device_ids=[0,1])
#model.to(device)

model = DataParallelModel(model, device_ids=[0, 1, 2])
model = model.to(device)

#print(model)
# when fintune the model use it
"""
for param in model.parameters():

    param.requires_grad = Falseparamer
"""
#rmse_loss = RMSE_Loss()
#dummy_input = torch.rand(4,10,1,480,480).to(device)
#with SummaryWriter(comment='convgru')as w:
    #w.add_graph(model,dummy_input)
##### load model parameters and dict ##########################################
###############################################################################
#changepoint  请改好模型参数的名字不然会冲突
if os.path.exists(os.path.join(save_dir, 'testcrossentropypredrnncheckpoint.pth.tar')) :
    # load existing model
    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'testcrossentropypredrnncheckpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch = model_info['epoch']
else:
    print('model parameters is not exist and build it')
    #input()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    optimizer = Ranger(model.parameters(), lr=args.lr,weight_decay=1e-4)
    # create model




#print(model)
for name in model.state_dict():
   print(name)

if isinstance(model,torch.nn.DataParallel):
	model = model.module
model_lastoutput = list(map(id, model.decoder.lastoutput.parameters()))

base_params = filter(lambda p: id(p) not in model_lastoutput,

                     model.parameters())

optimizersgd = torch.optim.SGD([

            {'params': base_params},

            {'params': model.decoder.lastoutput.parameters(), 'lr': 0.0001}], lr=0.00001, momentum=0.9)

optimizer = optimizersgd

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizersgd, T_max=10000, eta_min=0)

model = DataParallelModel(model, device_ids=[0, 1, 2])
model = model.to(device)

print('fuck off')

#optimizersgd = torch.optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9)
#################### chose your optimzer########################################
cur_epoch = 0
#################################################################################
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=0.7)
#################################################################################
                          #trian#
#################################################################################
                        #before the trian you must chose the criention 
#criterion = Weighted_mse_mae().to(device)  #(this loss is adapt for precipation)
#criterion = reinforce_loss()              #(this loss is use for normal model)
rmse_loss = RMSE_Loss()
criterionmse = nn.MSELoss()
criterionmae = nn.L1Loss()
criterionbmse = mixloass('hard')         #功利一点我认为这种比赛，用三种loss：bmsemae，时间系数mse，bcrossentropy,就可以了
criterionbmse = DataParallelCriterion(criterionbmse, device_ids=[0, 1, 2])
criterionbcrossentropy = WeightedCrossEntropyLoss(thresholds,weights,5)
criterionbcrossentropy = DataParallelCriterion(criterionbcrossentropy, device_ids=[0, 1, 2])

criteriondice = multiclassdice(thresholds,weights)
criteriondice = DataParallelCriterion(criteriondice,device_ids=[0,1,2])
#criterion2 = DataParallelCriterion(criterionbmse, device_ids=[0, 1, 2])
#################################################################################
writer = SummaryWriter(log_dir)
lossx=[]
rmsex =[]
early_stopping = EarlyStopping(patience=5, verbose=True)
val_draw =[]
step =0
lrr = args.lr
for epoch in tqdm(range(cur_epoch, args.epochs + 1)):
    rmse = AverageMeter()
    losses = AverageMeter()

    if epoch %2==0 and epoch >0 and False:
        lrr = lrr *0.5
        optimizer = adjust_learning_rate(optimizer, lrr)
    model.train()

    for headid in range(1):
        #break
        for ind, (_, (inputframes,targetframes)) in enumerate(train_loaders):
            #print(inputframes.shape)
            #exp_lr_scheduler.step()
            inputs = inputframes
            targets = targetframes
            ######################### 数 据 入 口 ######################################
            ############ the input shape  is b*s*c*w*h #################################
            #################  数据转换接口在这里########################################
            #################  请严格按照接口和出口处理数据，进入和出去的数据格式均为bscwh
            #因为要集成所以请务必严格遵守数据转换不然会很恶心不要问我怎么知道的，列如demo的unet的格式为bcwh
            #那么就在这里进行转换，c通道实际上相当于序列seq，这是一种很常见的处理方式#####
            #inputs = inputs.squeeze(2)
            #print(inputs.shape)
            ######################## 请在data转到gpu上前完成你的集合成员模型数据格式转换
            #############################################################################
            #############################################################################
            #############################################################################

            inputs = inputs.type(torch.FloatTensor).to(device)
            targets = targets.type(torch.FloatTensor)
            #print("the input size is :")
            #print(input.shape)
            #print("checkpoint begin")
            #print(inputx.shape)
            #print(target.shape)
            #print(inputx.shape)
            output = model(inputs)

            ############################ 数据出口#########################################
            ############  请将你的output数据重新转换成 bscwh格式进行下一步#################
            #output = output.unsqueeze(2)

            ###############################################################################
            #loss1 = criterionmse(output, targets)
            #loss2 = criterionmae(output, targets)
            print(targets.shape)
            loss_cross = criterionbcrossentropy(output,targets)
            loss_dice = criteriondice(output,targets)
            print('crossentropy loss --->',loss_cross)
            print('dice loss----->',loss_dice)
            loss = loss_cross 
            #loss3 = criteriongdl(output,target)
            #loss = loss1+loss2
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            #t_rmse = rmse_loss(output, targets)
            #rmse.update(t_rmse.item())  


            #output_np = np.clip(output.detach().cpu().numpy(), 0, 1)
            #target_np = np.clip(targets.detach().cpu().numpy(), 0, 1)
            #scheduler.step()
            torch.cuda.empty_cache()
            logging.info('[{0}][{1}][{2}]\t'
                'lr: {lr:.5f}\t'
                'loss: {loss.val:.6f} ({loss.avg:.6f})'.format(
                epoch, headid, ind, 
                lr=optimizer.param_groups[-1]['lr'],
                loss=losses
                ))
            writer.add_scalars("trainloss", {
                "train": losses.val
                }, step)
            step +=1

###############################################tenosrboard太麻烦######
            lossx.append(losses.val)
            #rmsex.append(rmse.val)
            x = range(len(lossx))
            plt.figure(1)
            plt.title("this is loss and rmse")
            plt.plot(x,lossx,label='loss')
            #plt.plot(x,rmsex,label='rmse')
            plt.legend()
            #changepoint 方便查看tensorboard太麻烦
            plt.savefig('/media/workdir/hujh/hujh-new/huaweirader_baseline/log/demolog/crossentropypredrnnloss.png')
            plt.close(1)
#################################################################################
            #changepoint
            if ind %100 ==0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}, 
                    save_dir=save_dir,
                    filename='fucktestcrossentropypredrnncheckpoint.pth.tar')
################# valid ########################################################
            #if ind % 1000 ==0 and ind > 0:
    val_compareloss = []
    hss = []
    model.eval()
    if True :
        with torch.no_grad():
            val_rmse = AverageMeter()
            val_losses = AverageMeter()
            
            if True:
                
                for headid in range(1):
                    #break
                    for ind, (_, (inputframes,targetframes)) in enumerate(valid_loaders):
                        val_inputs = inputframes
                        val_targets = targetframes  
                        #print(output)
                        #### 数据转换入口########
                        #val_inputs = val_inputs.squeeze(2)
                        #########################
                        val_inputs = val_inputs.type(torch.FloatTensor).to(device)
                        val_targets = val_targets.type(torch.FloatTensor).to(device)
                        print(val_inputs.shape)
                        #break
                        val_output = model(val_inputs)
                        #######数据转换出口######
                        #val_output = val_output.unsqueeze(2)
                        ########################
                        prob = ProbToPixel(midlevalue)
                        for i in range(len(val_output)):
                            val_output[i] = (prob(val_output[i][0]),)
                        val_loss = criterionbmse(val_output, val_targets)
                        print('val_loss------->',val_loss.item())
                        val_compareloss.append(val_loss.item())
                        #tt_rmse = rmse_loss(val_output, val_targets)
                        val_losses.update(val_loss.item())
                        #val_rmse.update(tt_rmse.item())
                        #output_np11 = np.clip(val_output.detach().cpu().numpy(), 0, 1)
                        #target_np11 = np.clip(val_targets.detach().cpu().numpy(), 0, 1)
                        ####### hss评分接口#################

                        ###################################
                        
                        writer.add_scalars("val_loss", {
                            "valid": val_losses.val
                            }, epoch)
        valid_loss = np.average(val_compareloss)
        val_draw.append(valid_loss)
        plt.figure(2)
        plt.title("this is val loss")
        plt.plot(val_draw,label='loss')
        plt.legend()
        #changepoint
        plt.savefig('/media/workdir/hujh/hujh-new/huaweirader_baseline/log/demolog/crossentropypredrnnval.png')
        plt.close(2)

        early_stopping(valid_loss, model)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print("passing the unity testing")
                        



    

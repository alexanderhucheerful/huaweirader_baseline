
#author :alexhu
#time: 2020.4.30
#欢迎各位大佬指点交流：qq-》2473992731
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import os
from matplotlib import colors
import cv2 as cv2
import sys
sys.path.append("..")





#from contetualloss.VGG_Model import VGG_Model
import torch.nn.functional as F
import copy
#import torchsnooper
#######

class RMSE_Loss(nn.Module):
    def __init__(self):
        super(RMSE_Loss,self).__init__()
    def forward(self, gen_frames, gt_frames):
        mse = F.mse_loss(gen_frames, gt_frames)
        return torch.sqrt(mse)

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))



class huber_loss(nn.Module):
    def __init__(self):
        super(huber_loss,self).__init__()
        huber = nn.SmoothL1Loss()
    def forward(self,input,output):
        return huber(input,output)

class mse_mae(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, output):
        return (torch.mean(torch.pow((x - y), 2))+torch.mean(torch.abs(x-y)))

class reinforce_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        w = torch.ones_like(x)
        w = torch.where(x<y,w*10,w*1)
        return (torch.mean(w*torch.pow((x - y), 2))+torch.mean(w*torch.abs(x-y)))

class mse_reinforce_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        w = torch.ones_like(x)
        w = torch.where(x>y,w*20,w*1)
        return (torch.mean(w*torch.pow((x - y), 2)))

class BLOSS(nn.Module):
    def __init__(self):
        super(BLOSS,self).__init__()
        self.w_1 = [0,1,2,3,4,5]
        self.y = [10,20,30,40,50,60]
        
    def forward(self,input,target):
        w =target.clone()
        for i in range(len(self.w_1)):
            w[w<self.y[i]] = self.w_l[i]

        return torch.mean(w*((input-target)**2))+torch.mean(w*torch.abs(input-target))

class time_reinforce_loss(nn.Module):
    def __init__(self):
        super(time_reinforce_loss,self).__init__()
        # 自权重系数随时间增加，更偏重后时刻的效果
        self.delt = 10.0/10
        self.weights = [(i+1)*self.delt for i in range(10)]
        
    def forward(self,input,target):
        weight = target.clone()
        w1 = torch.ones_like(x)
        w2 = torch.Tensor(self.weights)
        w = torch.Tensor(list(map(lambda w1,w2:w1*w2,w1,w2)))
        weight = w
        
        return torch.mean(w*((input-target)**2))+torch.mean(w*torch.abs(input-target))






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



    # input: output prob, S*B*C*H*W

    # target: S*B*1*H*W, original data, range [0, 1]

    # mask: S*B*1*H*W

    def forward(self, input, target, mask):

        assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN

        # F.cross_entropy should be B*C*S*H*W

        input = input.permute((1, 2, 0, 3, 4))

        # B*S*H*W

        target = target.permute((1, 2, 0, 3, 4)).squeeze(1)

        class_index = torch.zeros_like(target).long()

        thresholds = [0.0] + rainfall_to_pixel(self._thresholds).tolist()

        # print(thresholds)

        for i, threshold in enumerate(thresholds):

            class_index[target >= threshold] = i

        error = F.cross_entropy(input, class_index, self._weight, reduction='none')

        if self._lambda is not None:

            B, S, H, W = error.size()



            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)

            if torch.cuda.is_available():

                w = w.to(error.get_device())

                # B, H, W, S

            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # S*B*1*H*W

        error = error.permute(1, 0, 2, 3).unsqueeze(2)

        return torch.mean(error*mask.float())

"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
"""

class Loss():

    def __init__(self, args):

        super().__init__()

        self.weights = np.array([1., 2., 5., 10., 30.])

        self.value_list = np.array([0, 0.283, 0.353, 0.424, 0.565, 1])

        self.name = args.loss_function

        

        if args.loss_function.upper() == 'BMSE':

            self.loss = self._bmse

        if args.loss_function.upper() == 'BMAE':

            self.loss = self._bmae

        if args.loss_function.upper() == 'MSE':

            self.loss = self._mse

        if args.loss_function.upper() == 'MAE':

            self.loss = self._mae



    def _mse(self, x, y):

        return torch.sum((x-y)**2) / x.shape[0] / x.shape[1]



    def _mae(self, x, y):

        return torch.sum(torch.abs(x-y)) / x.shape[0] / x.shape[1]



    def _bmse(self, x, y):

        w = torch.clone(y)

        for i in range(len(self.weights)):

            w[w < self.value_list[i]] = self.weights[i]

        return torch.sum(w*((y-x)** 2)) / x.shape[0] / x.shape[1]



    def _bmae(self, x, y):

        w = torch.clone(y)

        for i in range(len(self.weights)):

            w[w < self.value_list[i]] = self.weights[i]

        return torch.sum(w*(abs(y - x))) / x.shape[0] / x.shape[1]



    def __call__ (self, outputs, targets):

        return self.loss(outputs, targets)



class LOSS_pytorch():

    def __init__(self, args):

        super().__init__()

        self.weights = np.array([1., 2., 5., 10., 30., 100.])

        self.value_list = np.array([0., 2., 5., 10., 30., 60., 200.])

        max_values = args.max_values['QPE']



        if args.target_RAD:

            max_values = args.max_values['RAD']

            self.value_list[1:] = R2DBZ(self.value_list[1:])



        if args.normalize_target:

            self.value_list = self.value_list / max_values

        

        if args.loss_function.upper() == 'BMSE':

            self.loss = mse

        if args.loss_function.upper() == 'BMAE':

            self.loss = mae

        if args.loss_function.upper() == 'MSE':

            self.loss = mse

            self.weights = np.ones_like(self.weights)

        if args.loss_function.upper() == 'MAE':

            self.loss = mae

            self.weights = np.ones_like(self.weights)


class Bmsemae(nn.Module):
    def __init__(self):
        super(Bmsemae,self).__init__()
        self.w_1 = [5,10,15,20,25,30]
        self.y = [0.16,0.33,0.5,0.66,0.83,1]
        
    def forward(self,input,target):
        w =target.clone()
        for i in range(len(self.w_1)):
            w[w<self.y[i]] = self.w_1[i]

        return torch.mean(w*((input-target)**2))+torch.mean(w*torch.abs(input-target))





def draw_video(output_np11,target_np11,colorbar,savepath,model_name,epoch,indx):
    """
    this def is draw the compare pic when process in the training or validating
    ------
    in: predarray,targetarray,pic_savepath,colorbar,the path your save,model's name,epoch,the number of valid
    ------
    return: the series video

    """
    output_np11 = output_np11
    target_np11 = target_np11
    colorbar = colorbar
    path = savepath
    model_name =model_name
    indx = indx
    epoch = epoch
    piclist = []



    for indp in range(output_np11.shape[2]): 
        temp1 = np.concatenate((
            np.transpose(target_np11[0, :, indp, :, :], axes=[1, 2, 0]), 
            np.transpose(output_np11[0, :, indp, :, :], axes=[1, 2, 0])
            ), axis=1)*60.0  # only show first output
        #temp1 = temp1*60.0
        #print(temp.shape)
        #np.squeeze(temp,axis=2)
        temp11 = np.zeros([500,1000])
        temp11 = temp1[:,:,0]
        #print(temp1.shape)
        plt.figure(2)
        plt.title('the epoch:'+str(epoch)+'valid_number:'+str(indx))
        plt.imshow(temp11,cmap=cmap_color)
        plt.colorbar()
        #plt.show()
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path,'beginning.png')) 
        pic = cv2.imread(os.path.join(path,'beginning.png'))[:,:,::-1]
        piclist.append(pic)
        plt.close(2)
    clip = mpy.ImageSequenceClip(piclist, with_mask=False, fps=2)
    clip.write_videofile(os.path.join(path, 'epoch:_%04d_validnumber:_%d.mp4'%(epoch,indx)), audio=False, verbose=False, threads=8)



################################ #context loss ###################333###############################################

"""

if __name__ == '__main__':

    from PIL import Image

    from torchvision.transforms import transforms

    import torch.nn.functional as F

    layers = {

            "conv_1_1": 1.0,

            "conv_3_2": 1.0

        }

    I = torch.rand(1, 3, 128, 128).cuda()

    T = torch.randn(1, 3, 128, 128).cuda()

    contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()

    print(contex_loss(I, T))

"""
    
class Distance_Type:

    L2_Distance = 0

    L1_Distance = 1

    Cosine_Distance = 2



"""

config file is a dict.

layers_weights: dict, e.g., {'conv_1_1': 1.0, 'conv_3_2': 1.0}

crop_quarter: boolean



"""

class Contextual_Loss(nn.Module):

    def __init__(self, layers_weights, crop_quarter=False, max_1d_size=100, distance_type=Distance_Type.Cosine_Distance, b=1.0, h=0.1, cuda=True):

        super(Contextual_Loss, self).__init__()

        listen_list = []

        self.layers_weights = {}

        try:

            listen_list = layers_weights.keys()

            self.layers_weights = layers_weights

        except:

            pass

        self.vgg_pred = VGG_Model(listen_list=listen_list)

        # self.vgg_gt = VGG_Model(listen_list=listen_list)

        # if cuda:

        #     self.vgg_pred = nn.DataParallel(self.vgg_pred.cuda())

            # self.vgg_gt = nn.DataParallel(self.vgg_gt.cuda())

        self.crop_quarter = crop_quarter

        self.distanceType = distance_type

        self.max_1d_size = max_1d_size

        self.b = b

        self.h = h





    def forward(self, images, gt):

        if images.device.type == 'cpu':

            loss = torch.zeros(1)

            vgg_images = self.vgg_pred(images)

            vgg_images = {k: v.clone() for k, v in vgg_images.items()}

            vgg_gt = self.vgg_pred(gt)

        else:

            id_cuda = torch.cuda.current_device()

            loss = torch.zeros(1).cuda(id_cuda)

            vgg_images = self.vgg_pred(images)

            vgg_images = {k: v.clone().cuda(id_cuda) for k, v in vgg_images.items()}

            vgg_gt = self.vgg_pred(gt)

            vgg_gt = {k: v.cuda(id_cuda) for k, v in vgg_gt.items()}

        # print('images', [v.device for k, v in vgg_images.items()])

        # print('gt', [v.device for k, v in vgg_gt.items()])



        for key in self.layers_weights.keys():

            N, C, H, W = vgg_images[key].size()



            if self.crop_quarter:

                vgg_images[key] = self._crop_quarters()



            if H*W > self.max_1d_size**2:

                vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)

                vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)



            loss_t = self.calculate_CX_Loss(vgg_images[key], vgg_gt[key])

            # print(loss_t)

            loss += loss_t * self.layers_weights[key]

            # del vgg_images[key], vgg_gt[key]

        return loss





    @staticmethod

    def _random_sampling(tensor, n, indices):

        N, C, H, W = tensor.size()

        S = H * W

        tensor = tensor.view(N, C, S)

        if indices is None:

            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()

            indices = indices.view(1, 1, -1).expand(N, C, -1)

        indices = Contextual_Loss._move_to_current_device(indices)



        # print('current_device', torch.cuda.current_device(), tensor.device, indices.device)

        res = torch.gather(tensor, index=indices, dim=-1)

        return res, indices



    @staticmethod

    def _move_to_current_device(tensor):

        if tensor.device.type == 'cuda':

            id = torch.cuda.current_device()

            return tensor.cuda(id)

        return tensor



    @staticmethod

    def _random_pooling(feats, output_1d_size=100):

        single_input = type(feats) is torch.Tensor



        if single_input:

            feats = [feats]



        N, C, H, W = feats[0].size()

        feats_sample, indices = Contextual_Loss._random_sampling(feats[0], output_1d_size**2, None)

        res = [feats_sample]



        for i in range(1, len(feats)):

            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)

            res.append(feats_sample)



        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]



        if single_input:

            return res[0]

        return res



    @staticmethod

    def _crop_quarters(feature):

        N, fC, fH, fW = feature.size()

        quarters_list = []

        quarters_list.append(feature[..., 0:round(fH / 2), 0:round(fW / 2)])

        quarters_list.append(feature[..., 0:round(fH / 2), round(fW / 2):])

        quarters_list.append(feature[..., round(fH / 2), 0:round(fW / 2)])

        quarters_list.append(feature[..., round(fH / 2):, round(fW / 2):])



        feature_tensor = torch.cat(quarters_list, dim=0)

        return feature_tensor



    @staticmethod

    def _create_using_L2(I_features, T_features):

        """

        Calculating the distance between each feature of I and T

        :param I_features:

        :param T_features:

        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position

        """

        assert I_features.size() == T_features.size()

        N, C, H, W = I_features.size()



        Ivecs = I_features.view(N, C, -1)

        Tvecs = T_features.view(N, C, -1)

        #

        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)

        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)

        # raw_distance

        raw_distance = []

        for i in range(N):

            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]

            # matrix multiplication

            AB = Ivec.permute(1, 0) @ Tvec

            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB



            raw_distance.append(dist.view(1, H, W, H*W))

        raw_distance = torch.cat(raw_distance, dim=0)

        raw_distance = torch.clamp(raw_distance, 0.0)

        return raw_distance



    @staticmethod

    def _create_using_L1(I_features, T_features):

        assert I_features.size() == T_features.size()

        N, C, H, W = I_features.size()



        Ivecs = I_features.view(N, C, -1)

        Tvecs = T_features.view(N, C, -1)



        raw_distance = []

        for i in range(N):

            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]

            dist = torch.sum(

                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False

            )

            raw_distance.append(dist.view(1, H, W, H*W))

        raw_distance = torch.cat(raw_distance, dim=0)

        return raw_distance



    @staticmethod

    def _centered_by_T(I, T):

        mean_T = T.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        # print(I.device, T.device, mean_T.device)

        return I-mean_T, T-mean_T



    @staticmethod

    def _normalized_L2_channelwise(tensor):

        norms = tensor.norm(p=2, dim=1, keepdim=True)

        return tensor / norms



    @staticmethod

    def _create_using_dotP(I_features, T_features):

        assert I_features.size() == T_features.size()

        I_features, T_features = Contextual_Loss._centered_by_T(I_features, T_features)

        I_features = Contextual_Loss._normalized_L2_channelwise(I_features)

        T_features = Contextual_Loss._normalized_L2_channelwise(T_features)



        N, C, H, W = I_features.size()

        cosine_dist = []

        for i in range(N):

            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous()

            I_features_i = I_features[i].unsqueeze(0)

            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()

            cosine_dist.append(dist)

        cosine_dist = torch.cat(cosine_dist, dim=0)

        cosine_dist = (1 - cosine_dist) / 2

        cosine_dist = cosine_dist.clamp(min=0.0)

        return cosine_dist





    @staticmethod

    def _calculate_relative_distance(raw_distance, epsilon=1e-5):

        """

        Normalizing the distances first as Eq. (2) in paper

        :param raw_distance:

        :param epsilon:

        :return:

        """

        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]

        relative_dist = raw_distance / (div + epsilon)

        return relative_dist



    def calculate_CX_Loss(self, I_features, T_features):

        I_features = Contextual_Loss._move_to_current_device(I_features)

        T_features = Contextual_Loss._move_to_current_device(T_features)



        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(torch.isinf(I_features)) == torch.numel(I_features):

            print(I_features)

            raise ValueError('NaN or Inf in I_features')

        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(

                torch.isinf(T_features)) == torch.numel(T_features):

            print(T_features)

            raise ValueError('NaN or Inf in T_features')



        if self.distanceType == Distance_Type.L1_Distance:

            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)

        elif self.distanceType == Distance_Type.L2_Distance:

            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)

        else:

            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)

        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(

                torch.isinf(raw_distance)) == torch.numel(raw_distance):

            print(raw_distance)

            raise ValueError('NaN or Inf in raw_distance')



        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)

        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(

                torch.isinf(relative_distance)) == torch.numel(relative_distance):

            print(relative_distance)

            raise ValueError('NaN or Inf in relative_distance')

        del raw_distance



        exp_distance = torch.exp((self.b - relative_distance) / self.h)

        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(

                torch.isinf(exp_distance)) == torch.numel(exp_distance):

            print(exp_distance)

            raise ValueError('NaN or Inf in exp_distance')

        del relative_distance

        # Similarity

        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)

        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(

                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):

            print(contextual_sim)

            raise ValueError('NaN or Inf in contextual_sim')

        del exp_distance

        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]

        del contextual_sim

        CS = torch.mean(max_gt_sim, dim=1)

        CX_loss = torch.mean(-torch.log(CS))

        if torch.isnan(CX_loss):

            raise ValueError('NaN in computing CX_loss')

        return CX_loss



######################## Normalization layers #############################3
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return myF.group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps
        )

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class SwitchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True,
                 using_bn=False, last_gamma=False):
        super(SwitchNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        if using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        if self.using_bn and (not self.using_moving_average):
            self.register_buffer('batch_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('batch_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.using_bn and (not self.using_moving_average):
            self.batch_mean.zero_()
            self.batch_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.batch_mean.add_(mean_bn.data)
                    self.batch_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

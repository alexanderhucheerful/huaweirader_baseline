import torch
#from nowcasting.config import cfg
import numpy as np
np.set_printoptions(threshold=np.inf)
#from trajgru.utils import *
from trajgru.trajGRU import wrap
from scipy.interpolate import NearestNDInterpolator
import sys
sys.path.append("..")
sys.path.insert(0, './VarFlow')
sys.path.append("/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/VarFlow/varflow")
from varflow import VarFlowFactory
#from nowcasting.hko.dataloader import precompute_mask
import matplotlib
matplotlib.use("Pdf")
from matplotlib import colors
import matplotlib.pyplot as plt


colorbar_dir = '/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/colorbar.txt'
rgb=np.loadtxt(colorbar_dir,delimiter=',')
rgb/=255.0
icmap=colors.ListedColormap(rgb,name='my_color')
cmap_color=icmap
norm = matplotlib.colors.Normalize(vmin=0,vmax=60)

def LastFrame(input):
    shape = list(input.shape)
    shape[0] = 10
    output = np.zeros(shape)
    for i in range(shape[0]):
        output[i, ...] = input[-1, ...]
    return output


class NonLinearRoverTransform(object):
    def __init__(self, Zc=33, sharpness=4):
        self.Zc = float(Zc)
        self.sharpness = float(sharpness)

    def transform(self, img):
        dbz_img = img*60.0
        dbz_lower = 0.0
        dbz_upper = 60.0
        transformed_lower = np.arctan((dbz_lower - self.Zc) / self.sharpness)
        transformed_upper = np.arctan((dbz_upper - self.Zc) / self.sharpness)
        transformed_img = np.arctan((dbz_img - self.Zc) / self.sharpness)
        transformed_img = (transformed_img - transformed_lower) / \
                          (transformed_upper - transformed_lower)
        return transformed_img

    def rev_transform(self, transformed_img):
        dbz_lower = 0.0
        dbz_upper = 60.0
        transformed_lower = np.arctan((dbz_lower - self.Zc) / self.sharpness)
        transformed_upper = np.arctan((dbz_upper - self.Zc) / self.sharpness)
        img = transformed_img * (transformed_upper - transformed_lower) + transformed_lower
        img = np.tan(img) * self.sharpness + self.Zc
        img = img/60.0
        return img


def nearest_neighbor_advection(im, flow):
    """

    Parameters
    ----------
    im : np.ndarray
        Shape: (batch_size, C, H, W)
    flow : np.ndarray
        Shape: (batch_size, 2, H, W)
    Returns
    -------
    new_im : nd.NDArray
    """
    predict_frame = np.empty(im.shape, dtype=im.dtype)
    batch_size, channel_num, height, width = im.shape
    assert channel_num == 1
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    interp_grid = np.hstack([grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1))])
    for i in range(batch_size):
        flow_interpolator = NearestNDInterpolator(interp_grid, im[i].ravel())
        predict_grid = interp_grid + np.hstack([flow[i][0].reshape((-1, 1)),
                                                flow[i][1].reshape((-1, 1))])
        predict_frame[i, 0, ...] = flow_interpolator(predict_grid).reshape((height, width))
    return predict_frame

class Rover(object):
    def __init__(self):
        self.transformer = NonLinearRoverTransform()
        self.flow_factory = VarFlowFactory(max_level=6, start_level=0,
                                  n1=2, n2=2,
                                  rho=1.5, alpha=2000,
                                  sigma=4.5)

    def __call__(self, input):
        prediction = np.zeros(shape=(10,) + input.shape[1:],
                              dtype=np.float32)
        I1 = input[-2, :, 0, :, :]
        I2 = input[-1, :, 0, :, :]
        mask_I1 = 1
        mask_I2 = 1
        I1 = I1 * mask_I1
        I2 = I2 * mask_I2
        I1 = self.transformer.transform(I1)
        I2 = self.transformer.transform(I2)
        flow = self.flow_factory.batch_calc_flow(I1=I1, I2=I2)
        #print(flow)
        if (I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2]))==I2[:,np.newaxis,:,:]).all():
            print("shape is ok")
        x = I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2]))
        """
        plt.imshow(x[0][0]*60.0,cmap=cmap_color,norm=norm)
        plt.show()
        plt.savefig('/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/train_benchmark/test.png')
        #input()
        """
        init_im = torch.from_numpy(I2.reshape((I2.shape[0], 1, I2.shape[1], I2.shape[2]))).cuda(2)
        
        nd_flow = torch.from_numpy(np.concatenate((flow[:, :1, :, :], -flow[:, 1:, :, :]), axis=1)).cuda(2)
        nd_pred_im = torch.zeros(prediction.shape)
        for i in range(10):
            new_im = wrap(init_im, -nd_flow)
            nd_pred_im[i][:] = new_im
            init_im[:] = new_im
        prediction = nd_pred_im.numpy()

        prediction = self.transformer.rev_transform(prediction)
        print(prediction.shape)
        #print(prediction[0][0][0])
        #plt.imshow(prediction[0][0][0]*60,cmap=cmap_color,norm=norm)
        #plt.show()
        #plt.savefig('/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/train_benchmark/test2.png')
        #sys.exit(0)##########################################################  fuckpoint
        return prediction




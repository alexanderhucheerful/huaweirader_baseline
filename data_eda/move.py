
import torch
import os
import pandas as pd
import struct
import numpy as np
import time
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import shutil
import scipy.misc

def default_loader_radar(path):
    pic = np.array(Image.open(path))

    trans = transforms.Compose([transforms.ToTensor()])
    pic = trans(pic)
    # print(pic)
    #
    #
    # pic=transforms.Resize(128)(transforms.ToPILImage()(pic))

    return pic


class MyDataSet(Dataset):
    # 初始化过程
    def __init__(self, dirpath, loader_radar=default_loader_radar,
                 ):
        self.dirpath = dirpath

        # 给雷达文件夹路径，因为txt文件和radar文件名没关系

        self.loader_radar = loader_radar
        self.path=[]
        self.mysamples = []
        pa = os.listdir(dirpath)
        for pat in pa:
            self.mysamples.append(os.path.join(dirpath, pat))
            self.path.append(pat)
    def __getitem__(self, index):
        # 将从文件中读取的数据返回到该列表
        picpath=self.path[index]
        dirname = self.mysamples[index]

        datas = os.listdir(dirname)
        datas.sort(key=lambda x: int(x[-7:-4]))
        tup = []
        for data in datas:

            adata = os.path.join(dirname, data)
            tup.append(adata)

        radars = []

        # 把每一个文件加载进来
        for name in tup:

            labels_read = self.loader_radar(name)
            radars.append(labels_read)
        radar_ = torch.stack(radars, dim=0)

        return radar_, dirname,picpath

    def __len__(self):
        return len(self.mysamples)


if __name__ == '__main__':
    txtpath = r'C:\Users\office\Desktop/Testnew'
    rader_path = r'/workspace/stpre/radarShenZhen/radarShenzhen'
    train_my_dataset = MyDataSet(dirpath=txtpath

                                 )

    test_my_dataset = MyDataSet(dirpath=txtpath
                                )

    # --------------- 说明 -----------------------
    # num_workers - 0为单线程, -1为全部线程
    # drop_last - True为舍弃最后不能并入batch的多余训练数据
    # pin_memory - 设置为 True， DataLoader 会在返回之前将 tensor 复制到 CUDA 的固定内存中
    # -------------------------------------------

    # train_db, val_db = torch.utils.data.random_split(train_my_dataset, [160, 40])
    trainloader = torch.utils.data.DataLoader(train_my_dataset,  # 将dataset封装成trainloader,赋予index等属性
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,

                                              )
    testloader = torch.utils.data.DataLoader(test_my_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,

                                             )
    print(len(trainloader))
    # trainset, testset = sklearn.train_test_split(trainloader, test_size=0.2)
    c = 0
    for a, b ,d in trainloader:
        tf = False

        for i in range(0, 21):

            if a[0, i].mean() == 0:

                print(str(b) + "No:" + str(i))
                if (i==0 or i==2 or i==4 or i==6 or i==8 or i==10 or i==11 or i==12 or i==13 or i==15 or i==17 or i==19 or i==14 or i==16 or i==18 or i==20):
                    #这里是有用到的图片序号，根据自己的去改
                    tf = True





        if tf == True:
            c += 1
            print(c)
            shutil.move(b[0], r"C:\Users\office\Desktop/Testmiss")

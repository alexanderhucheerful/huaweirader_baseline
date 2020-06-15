# -*- coding: utf-8 -*-

import numpy as np
import struct
import xarray as xr
import os
import math
import datetime
from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import cv2
from datasets import processTestloadedDataset
from multiprocessing import Pool
import torch 
import scipy.misc
import PIL.Image as Image
from rover_and_last_frame import Rover


def statistics(filename,data,threshold = 0.4):
    """
    data is dict 
    dict key :filename
    dict value: data-->(predpicdata,beforeinputframe(90min,120min))
    -------
    if predpicdata/beforedata <0.3 we think it predict false need use the rover to fillnan

    """
    caculateoptflow_data = data[filename][1]
    caculateoptflow_data = caculateoptflow_data/80.0
    predict_data = data[filename][0]
    #print(predict_data.shape)
    predict_data[predict_data<20] = 0.0
    #predict_data = predict_data/80.0
    # decision which data need to cover
    #print(caculateoptflow_data.shape)
    #print(caculateoptflow_data[-1].sum())
    #print(predict_data.sum())
    for ind,data in enumerate(predict_data):
        beginchange_number = ind
        if ind ==0:
            judge = data.mean()/(caculateoptflow_data[-1].mean()+0.0000000000001) 
        else:
            judge = data.mean()/(predict_data[ind-1].mean()+0.0000000000001) 
        if judge <threshold:
            break
    """
    if judge<0.4:
        print(filename)
        return filename
    """
    #print(judge)
    if judge < threshold or True:
        if True:
            print('begin the changing---------')
            model = Rover()
            caculateoptflow_data = caculateoptflow_data[:,np.newaxis,:,:,:]
            print(caculateoptflow_data[0].sum()-caculateoptflow_data[1].sum())
            if (caculateoptflow_data[0] == caculateoptflow_data[1]).all():
                print("that caculateoptflow data is break-------------------------")
            predict = model(caculateoptflow_data)
            predict = predict*80.0
            print('predict diff------->',predict[0].sum() - predict[1].sum())
            if (predict[0] == predict[1]).all():
                print("that pred data is break-------------------------")
            #print(predict.shape)
            timefilename = [30,60,90,120]
            for ind,time in enumerate(timefilename) :
                picname = str(time)+'.png'
                pic_savepath = os.path.join('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/Predict/fuck',filename)
                #pic_savepath = os.path.join(savepath)
                if  not os.path.exists(pic_savepath) :
                    os.mkdir(pic_savepath)
                pic_savepath = os.path.join('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/Predict/fuck',filename,picname)
                #scipy.misc.toimage(predict_np, high=255, low=0, cmin=0, cmax=255).save(pic_savepath)
                #predict_np_save = predict_np[time,:,:]
                #print(predict_np_save.shape)
                #print(predict[ind][0].shape)
                if ind >= beginchange_number or True:
                    #os.remove(pic_savepath)
                    Image.fromarray(np.uint8(predict[ind*2+2][0][0])).save(pic_savepath)
            print('already change the data')

        return filename


    #return filename
    #pass

def openpic(picpath):
    """
    open the predictpic path for multiprocess 

    """
    fourdata = []
    timefilename = [30,60,90,120]
    for time in timefilename:    
        processpath = os.path.join(dirpath,'Predict','Predict',picpath,str(time)+'.png') 
        data = np.array(Image.open(processpath))
        fourdata.append(data)
    fourdata = np.stack(fourdata)
    return (picpath,fourdata)



if __name__ == '__main__':

    dirpath = os.getcwd()
    test_npy_path  = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/test_demo.npy'
    valid_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/test_all_pkl.pkl'
    valid_dir = ''
    valid_loaders = []
    all_datasets = processTestloadedDataset(valid_dir, valid_path,test_npy_path  )
    valid_loaders =  torch.utils.data.DataLoader(all_datasets,batch_size  =8,shuffle=False)
    
    all_list = []
    for ind, (filename, inputframes) in enumerate(valid_loaders):
        print(ind)
        temp_list = []
        inputframes = inputframes.numpy()[:,:,:]*80.0
        print('inputframes---shape--->',inputframes.shape)
        inputframes = dict(list(zip(filename,inputframes)))
        #print(inputframes.shape)
        with Pool(8) as p:
            picdata = dict(list(p.map(openpic, filename)))
        #picdata = np.stack(picdata)
        #print(picdata.shape)------>(batchsize,256,256)
        #zipdata = list(zip(inputframes,picdata))
        #print(zipdata[0][1].shape)------>[(inputframe1,picdata1),(inputframe2,picdata2),...(inputframen,picdatan)]
        fusionzipdata ={}
        for key,_ in picdata.items():
            fusionzipdata[key] = (picdata[key],inputframes[key])
        #print(fusionzipdata[list(fusionzipdata.keys())[0]][0].shape)
        p = Pool(8)
        #multiprocess the data for save my time 
        for key in fusionzipdata.keys():
            name = p.apply_async(statistics,args=(key,fusionzipdata))
            temp_list.append(name)
        p.close()
        p.join()
        temp = [x.get() for x in temp_list]
        #print(temp)
        with open(os.path.join(dirpath,'havesamequesion.txt'),"a") as file: 
            for filename in temp:
                if filename is not None:
                    file.write(filename + "\n")
        #print(temp)
        all_list.extend(temp)


        
        #poordata = list(p.map(lambda (k,v):statistics(k,v),fusionzipdata.items()))
        #all_list.extend()

        






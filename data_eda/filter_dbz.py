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
import numba
from pathos.multiprocessing import ProcessingPool as Pool



class filtrate_dbz():
    def __init__(self,pd_path = None,rootdir = None,area=256*256,threshold=20,frame_num=20,time_freq=1):
        #the criterion is decide in this:
        #the pd path is your pkl location  root_dir is your png loaction
        #the area is 30*30 that they must count >900 10dbz uper
        self.trainpd_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl'
        self.testpd_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/test_all_pkl.pkl'
        self.rootdir = ''
        self.area = area
        self.threshold = threshold
        self.rootdir = rootdir
        self.fnum = frame_num
        self.freq = time_freq
        self.train_df = pd.read_pickle(trainpd_path)
        self.test_df = pd.read_pickle(testpd_path)
        # 上保险
        temp_df = self._df
        self._df = self._df.sort_values(by=['fname'])
        assert (self._df.values == temp_df.values).all()
        #清洗数据
        index = list(map(lambda x:x[0].split('/')[-2],self.train_df.values))
        self.train_df.index = [index,index]
        self.train_df.index.names=['index0','index1']

        index = list(map(lambda x:x[0].split('/')[-2],self.test_df.values))
        self.test_df.index = [index,index]
        self.test_df.index.names=['index0','index1']

        """
        this is my process in jupyter filter that one is not complete so i drop it 
        ----


        """


    def read_pic(self,tid):
        tid = tid[0]
        frame = np.array(Image.open(os.path.join(self.rootdir, self._df['fname'][tid])))
        frame[frame>=80.5] = 0
        #print(frame)
        count  = (frame > self.threshold)
        #print(self.threshold)
        counts = np.logical_and(count,True).sum()
        print(tid,'is processing')
        if counts >=self.area:
            x = tid
        else:
            x= None
        return x

    #@numba.jit
    def return_index(self):
        #index_list = []
        print("begin map")
        #this  has sonme bugs fix the bug
        #index_list = list(map(lambda x:self.read_pic(x),self.index))
        print(" it will   god bless no fuck bug")
        def splitstr_int(var):
            if var[0].split('/')[-1].split('.')[0][-2] == 0:
                var = int(var[0].split('/')[-1].split('.')[0][-1:])
            else:
                var = int(var[0].split('/')[-1].split('.')[0][-2:])
            return var
        number_list = list(map(splitstr_int,self._df.values))
        number_list = list(filter(lambda x:x[1]<self.eachsetlen,enumerate(number_list)))
        print(number_list)
        
        with Pool(8) as p:
            x=list(p.map(self.read_pic, number_list))
        index_list = list(filter(lambda x:x!=None,x))
        
        
        """
        for i in self.index:
            x = self.read_pic(i)
            if x != None:      
                print(x)
                #time.sleep(1)
                index_list.append(x)
        print("that ok")
        return index_list
        """
        print("select the train and test index is ok")
        return index_list

if __name__ == '__main__':
    pd_path = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl'
    root_dir = ''
    filtrate_dbz = filtrate_dbz(pd_path=pd_path,rootdir=root_dir)
    list_number = filtrate_dbz.return_index()
    list_number = np.array(list_number)
    save_path = "/media/workdir/hujh/hujh-new/rader-baseline-alexhumaster/train_benchmark/trian_filterdbz.npy"
    x = np.save(save_path,list_number)

    
		

        

        

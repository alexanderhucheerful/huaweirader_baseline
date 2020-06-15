import PIL.Image as Image
import os
import numpy as np 
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
def makefeatrue(tid,pkl_path='/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl',save='/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_distribute.npy'):
    """
    reutrn[seq_dbz_mean,seqmax,seqmin,seqfinally-seqbegin,seq_begin,target_seq_mean]
    """
    tid = tid
    #print(tid)
    df = pd.read_pickle(pkl_path)
    rootdir = ''
    count = []
    seq_mean = []
    target_seq_mean = []
    nan_value = []
    for time in range(tid,tid+22,2):
        frame = np.array(Image.open(df['fname'][time]))
        frame[frame>85] = 0
        temp = frame.sum()
        if time == tid:
            seq_begin = temp
        if time ==tid+20:
            seq_finally = temp
        seq_mean.append(temp/(np.sum(frame>0.0000001)))
        count.append(np.sum(frame>0.000001))
    if True:
        for time in range(tid+25,tid+45,5):
            frame = np.array(Image.open(df['fname'][time]))
            frame[frame>85] = 0
            print(np.sum(frame>0.01))
            if np.sum(frame>0.01) ==0:
                nan_value.append(df['fname'][time])
            target_seq_mean.append(np.sum(frame)/(np.sum(frame>0.000001)))
    seq_max = np.array(seq_mean).max()
    seq_min = np.array(seq_mean).min()
    seq_mean = np.array(seq_mean).mean()
    seq_count = np.array(np.array(count).mean())
    anomaly = int(seq_finally) - int(seq_begin)
    #print(anomaly)
    target_seq_mean = np.array(target_seq_mean).mean()
    nanfile = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/nan.txt'
    with open(nanfile,"a") as file: 
        for filename in nan_value:
            file.write(filename+ " "+"\n")
    return np.array([seq_mean,seq_max,seq_min,anomaly,seq_begin,seq_finally,seq_count,target_seq_mean])

x = np.load('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_demo.npy')
x = list(x)
print("begin....")
with Pool(8) as p:
    #train_list=list(p.map(demo_pic,train_number_list))
    test_list = list(p.map(makefeatrue,x))
test_list = np.array(test_list)
np.save('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_distrubute.npy',test_list)
print("finish")

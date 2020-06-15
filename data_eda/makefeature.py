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
    print(tid)
    df = pd.read_pickle(pkl_path)
    rootdir = ''
    seq_mean = []
    target_seq_mean = []
    for time in range(tid,tid+22,2):
        frame = np.array(Image.open(df['fname'][time]))
        temp = frame.sum()
        if time == tid:
            seq_begin = temp
        if time ==tid+20:
            seq_finally = temp
        seq_mean.append(temp)
    if True:
        for time in range(tid+25,tid+45,5):
            frame = np.array(Image.open(df['fname'][time]))
            target_seq_mean.append(np.sum(frame)/(256*256.0-np.sum(frame<0.01)))
    seq_max = np.array(seq_mean).max()
    seq_min = np.array(seq_mean).min()
    seq_mean = np.array(seq_mean).mean()
    anomaly = int(seq_finally) - int(seq_begin)
    print(anomaly)
    target_seq_mean = np.array(target_seq_mean).mean()
    return np.array([seq_mean,seq_max,seq_min,anomaly,seq_begin,seq_finally,target_seq_mean])

x = np.load('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_demo.npy')
x = list(x)
with Pool(8) as p:
    #train_list=list(p.map(demo_pic,train_number_list))
    test_list = list(p.map(makefeatrue,x))
test_list = np.array(test_list)
np.save('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_distrubute.npy',test_list)

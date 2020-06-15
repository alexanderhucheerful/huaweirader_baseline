import PIL.Image as Image
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import os
def filter_hard(tid):
    df = pd.read_pickle('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl')
    tid = tid
    print(tid)
    seq_mean = []
    for time in range(tid+1,tid+20,2):
        frame = np.array(Image.open(df['fname'][time]))
        #缺测值为255我是直接将其滤成0不知道有没有其他好一点的办法，或者random一个0-80dbz的数值
        temp = frame.sum()
        frame[frame>=85.5] = 0
        seq_mean.append(temp/(np.sum(frame>0.0000001)))
    #print(frame)
    """
    for time in range(tid+1,tid+20,2):
    count  = (frame > 20)
    #print(self.threshold)
    counts = np.logical_and(count,True).sum()
    print(tid,'is processing')
    if counts >=self.area:
        x = tid
    else:
        x= None
    """
    seq_mean = np.array(seq_mean).mean()
    if seq_mean >=24:
        return tid
    else:
        return None
    

#demo不进行任何操作直接返回
def demo_pic(tid):
    return tid[0]
test_number_listx = np.load('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_demo.npy')
test_number_list = test_number_listx.tolist()
print("begin")
with Pool(16) as p:
    #train_list=list(p.map(demo_pic,train_number_list))
    test_list = list(p.map(filter_hard,test_number_list))
#train_list = np.array(train_list)
print("over")
test_list  = np.array(test_list)
#这样返回的索引就是筛选过后的
#index_list = list(filter(lambda x:x!=None,x))
#train_savepath = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_demo.npy'
test_savepath = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/hard_train_demo.npy'
#save = np.save(train_savepath,train_list)
save = np.save(test_savepath,test_list)
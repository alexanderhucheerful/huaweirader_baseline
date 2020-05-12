import pandas as pd
from multiprocessing import Pool
import PIL.Image as Image
import numpy as np

#重新制索引用来聚类分析
def reindex(dataframe):
    new_index = list(map(lambda x:x[0].split('/')[-2],dataframe.values))
    dataframe.index = [new_index,new_index]
    dataframe.index.names=['index0','index1']
    return dataframe

#分割获得每张图片在样本中的时序
def splitstr_int(var):
    if var[0].split('/')[-1].split('.')[0][-2] == 0:
        var = int(var[0].split('/')[-1].split('.')[0][-1:])
    else:
        var = int(var[0].split('/')[-1].split('.')[0][-2:])
    return var

def fliter_dif_ele(tid):
    print(tid)
    df = train_df
    for itid in tid:
        # print(df.index[tid])
        frame_0 = np.array(Image.open(df['fname'][itid[0]]))
        count_bf = np.sum(frame_0 == 0)
        diff = []
        for time in range(itid[0]+1,itid[0]+41):
            frame = np.array(Image.open(df['fname'][time]))
            count_af = np.sum(frame == 0)
            diff.append(np.abs(count_af-count_bf))
            count_bf = count_af

        threshold=2.5
        mean_d = np.mean(diff)
        std_d = np.std(diff)
        z_score = (diff - mean_d)/std_d

        for i in range(len(diff)-1):
            if np.abs(z_score[i]) > threshold and np.abs(z_score[i-1]) > threshold:
                # print(df.index[i+itid[0]])
                # print(i)
                frame = (np.array(Image.open(df['fname'][itid+i-1])) + np.array(Image.open(df['fname'][itid+i+1])))/2.0
                Image.fromarray(np.uint8(frame)).save(df['fname'][itid+i])

def main():
    train_pd = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl'
    test_pd = '/work1/home/wuzhiwen/huawei/huaweirader_baseline/data_eda/test_all_pkl.pkl'
    global train_df
    
    train_df = pd.read_pickle(train_pd)
    #test_df = pd.read_pickle(test_pd)
    """
    #二次检查排序方式
    checkorigan_df = train_df
    checksort_df = train_df.sort_values(by=['fname'])
    assert (checkorigan_df.values == checksort_df .values).all()
    train_df = reindex(train_df)
    test_df = reindex(test_df)
    print(train_df.describe())
    train_number_list = list(map(splitstr_int,train_df.values))
    test_number_list = list(map(splitstr_int,test_df.values))
    #因为一个训练样本包含41个图片时长为4h，考虑到一个seq就需要4h所以无法在样本内进行滑窗切割数据集，只需要返回初始下标即可，即x[1]==0
    train_number_list = list(filter(lambda x:x[1]==0,enumerate(train_number_list)))
    test_number_list = list(filter(lambda x:x[1]==0,enumerate(test_number_list)))
    # print(train_number_list[0][0])
    """
    train_number_list = np.load('/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_demo.npy')
    train_number_list = list(train_number_list)
    #with Pool(8) as p:
    fliter_dif_ele(train_number_list)


if __name__ == "__main__":
    print("beging")
    main()
    print("fininshing")
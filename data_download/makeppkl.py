import glob
import os
import pandas as pd
import pickle

#索引将以dataframe数据结构储存在pkl用来eda
train_pkl = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/train_all_pkl.pkl'
test_pkl = '/media/workdir/hujh/hujh-new/huaweirader_baseline/data_eda/test_all_pkl.pkl'
# unzip data file
train_unzip_path = "/media/data/huaweiraderdata/train/train_all/"
test_unzip_path = "/media/data/huaweiraderdata/test/test_all/"
train_raderlist = os.listdir(train_unzip_path)
test_raderlist = os.listdir(test_unzip_path)
train_raderlist.sort()
test_raderlist.sort()
print('traindata_len',len(train_raderlist))
print('testdata_len',len(test_raderlist))

train_datafile = []
test_datafile = []

for trainfile in train_raderlist:
    pic = glob.glob(os.path.join(train_unzip_path + trainfile + '/','**.png'))
    pic.sort()
    # check if it Satisfy number of 41 if is not drop it
    if len(pic) == 41:
        train_datafile.append(pic)
    else:
        pass

for testfile in test_raderlist:
    pic = glob.glob(os.path.join(test_unzip_path + testfile + '/','**.png'))
    pic.sort()
    # check if it Satisfy number of 41 if is not drop it    
    test_datafile.append(pic)

#convert the file to flatten list so it can convert to dataframe
train_numlist = [train_pic for filepath in train_datafile for train_pic in filepath]
test_numlist = [test_pic for filepath in test_datafile for test_pic in filepath]

print('train_pic number----->',len(train_numlist))
print('test_pic number----->',len(test_numlist))

output_train = open(train_pkl,'wb')
output_test = open(test_pkl,'wb')

df_train = pd.DataFrame(data=train_numlist,columns = ['fname'])
df_test = pd.DataFrame(data=test_numlist,columns = ['fname'])

pickle.dump(df_train,output_train)
pickle.dump(df_test,output_test)

output_train.close()
output_test.close()

print("process is ok")

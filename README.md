# huaweirader_baseline
this is a huaweirader_baseline

# 文档详细说明
## train&inference 流程

——————————
特征工程
1.首先是下载数据集
通过data_download文件下的data_download.sh脚本下载数据集，可能有点bug，需要手动检查

2.制作数据集文件索引
通过makeppkl.py脚本制作训练数据索引以dataframe格式保存在data_eda文件下的pkl文件中

3.数据分析挖掘分形以及索引重构
在data_eda.ipynb中进行雷达数据分析挖掘，分类后的训练索引以npy保存注意自定义的pytorch中datasets加载时，
不仅仅加载pkl文件，同样加载npy，因为是demo未作数据分型索引，train_demo.npy文件保留最原始的pkl中的索引
数据分析后，分型函数请封装好，以模块的形式嵌入filter_dbz.py的类中，或者在data_eda.ipynb处理也可以。




——————————
训练与inference
4.数据加载
在自定义的datasets.py中返回tensor

5.模型训练
在train_demo_unet.py中完成训练验证(是否可以考虑k-flod交叉验证以及模型参数保存

6.集成预测
在test_benchmark.py中完成集成预测

7.模型保存
model文件中保存模型block
model_parameters文件中保存标准封装结构的模型参数以及训练信息






——————————
如何加入集合成员模型
请按如下流程并入
1.在model文件中保存高度封装好的模型文件
2.在model_parameters文件中提交预训练模型
3.在主目录中提交训练程序，请严格按照指定位置数据接口进行tensor的shape转换









————————
train_demo_unet.py文档说明

#############文档说明############
#标准数据结构从datasets返回为 batchsize*seq*channel*width*height,请十分注意，且精度是doubletensor需要转化为floattensor
#直觉告诉我unet会在这个任务里效果显著
#可以尝试的方向有 predrnn++，mim，e3d，selfatten_convgru等集合成员
#本次训练采用early_stoping策略，ranger优化器，lr2个epoch衰减0.7
#训练测试集比为8：2随机划分
"""
本demo模板需要改动的路径如下:
1.各种储存路径
2.模型参数checkpoint的名称
3.数据接口，请务必高度封装你的模型，如encoder-decoer写成一个class不要散着写，可以直接导入train.py中
我将在每个需要改动的的地方插入changepoint断点请在ide里直接顺序find：changepoint
"""
###################################################






__________________
test_benchmark.py文档说明
#####################
"""
文档说明
同train.py
测试时顺序加载数据，batchsize为1，不shuffle
请同样严格遵循数据接口
需要改路径或名称的地方将插入changepoint，顺序搜索即可

"""
#####################


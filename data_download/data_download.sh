#define your datasaving path
#这个脚本有点bug谨慎运行
yourpath="../../"

# download the data 
wget https://ma-competitions-bj4ther.zip 

#我觉得制作这个数据集的人是故意的，脑子有病吧
#还得安装rar解压
wget http://www.rarlab.com/rar/rarlinux-x64-4.2.0.tar.gz
sudo su
tar zxvf rarlinux-x64-4.2.0.tar.gz -C /usr/local
ln -s /usr/local/rar/rar /usr/local/bin/rar
ln -s /usr/local/rar/unrar /usr/local/bin/unrar

#unzip file
unzip datasets_weather.zip

#make dir
mkdir  ${yourpath}/train
mkdir ${yourpath}/test

#The organizer is a fool the datasets name actually have chinese so it cant to unzip
# need to change the dataname to unzip
#vim -e U8bad#U7ec3#U96c61-2-3-4.zip<<-! 

unzip 训练集1-2-3-4.zip -d  ${yourpath}/train
unrar x TEST1-初赛测试集.rar -d  ${yourpath}/test

cd ${yourpath}/train 

mkdir train


unrar *1.rar ${yourpath}/train/train 
unrar *2.rar ${yourpath}/train/train 
unrar *3.rar ${yourpath}/train/train 
unrar *4.rar ${yourpath}/train/train 

#cd ${yourpath}/test
#ls


# --utf8--#
from os.path import join
import numpy
import random

class Argument():
    def __init__(self):

        self.max_log_scale = 2  #模型最大扩大多少倍   如果max_log_scale 为2 是指放大4倍
                                # 如果max_log_scale 为3 是指放大8倍   （在没有这个参数之前默认就是8x模型）


        self.input_channel = 3   #读取图片的通道数量
        self.conv_f = 3          #卷基层的边长
        self.conv_ft = 4         #反卷基层的边长
        self.conv_n = 64         #每一层的通道数（特征个数）
        self.depth = 7           #金字塔每一层深度学习的深度
        self.output_channel = 3  #重建层输出的通道数
        self.height = 248        #输入图像的高度
        self.width = 248         #输入输入的宽度
        self.decay = 2          #衰减速率
        self.decay_step = 50    #多少步学习速率衰减一次
        self.lr = 1e-4          #初始化的学习速率
        self.min_lr = 1e-30     #衰减到多少为止
        self.iter_nums = 1000   #迭代次数
        self.momentum = 0.9     #采用动量算法的动量
        self.test_epoches = 20  #测试的迭代周期数
        self.weight_decay = 1e-4  #权值衰减 (用tensorfow 自己实现的)
        self.train_data_path = "./dataset/BSDS300/images/train/"  #训练集文件夹的位置
        self.validation_data_path = "./dataset/BSDS300/images/validation"  # 验证集文件夹的位置
        self.test_data_path = "./dataset/BSDS300/images/test"  # 测试集文件夹的位置
        self.save_path = "./"   #保存模型参数的地方
        self.model_name = "testModel15"  #训练模型的名字


        """
            新训练集部分
        """
        self.flicker = True    # 新训练集是否打开
        self.mirflicker_dir = '/Users/wshwbluebird/ML/mirflickr/'  # mirflicker 训练集文件夹
        self.flicker_train_opt = 'xxx'  #mk训练集训练方式  random 为随机  如果不是random 就是按约定顺序
        self.flicker_file_index = 1  #mk训练集训练方式  连续型的训练指针
        self.flicker_begin_index = 1  # mk训练集训练方式  连续型的训练指针
        self.flicker_end_index = 25000  # mk训练集训练方式  连续型的训练指针

        self.batch_size = 16  # 每批训练数据的大小
        self.num_threads = 4  # 数据导入开启的线程数量
        self.min_after_dequeue = 1024  # 保证线程中至少剩下的数据数量
        self.flicker_random_list = list(numpy.arange(self.flicker_begin_index
                                                ,1+self.flicker_end_index,1))

    def predict(self,batchsize):
        self.batch_size = batchsize

    def get_file_list(self):
        toFileName  =lambda x: join(self.mirflicker_dir,'im'+str(x)+'.jpg')
        if options.flicker_train_opt == 'random':
            numList =  random.sample(options.flicker_random_list, options.batch_size)
            return list(map(toFileName,numList))
        else:
            filelist = []
            for i in range(self.batch_size):
                filename = join(self.mirflicker_dir,'im'+str(self.flicker_file_index)+'.jpg')
                filelist.append(filename)
                self.flicker_file_index = self.flicker_begin_index-1\
                                          +(self.flicker_file_index+1)%\
                                           (self.flicker_end_index-self.flicker_begin_index +1)
            return filelist





options = Argument()

if __name__ =='__main__':
    for i in range(10):
        print(options.get_file_list())
        print()



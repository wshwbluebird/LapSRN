# --utf8--#
from os.path import join
import numpy as np
import random
from PIL import Image

class Argument():
    def __init__(self):

        self.max_log_scale = 2  #模型最大扩大多少倍   如果max_log_scale 为2 是指放大4倍
                                # 如果max_log_scale 为3 是指放大8倍   （在没有这个参数之前默认就是8x模型）


        self.input_channel = 1   #读取图片的通道数量
        self.conv_f = 3          #卷基层的边长
        self.conv_ft = 4         #反卷基层的边长
        self.conv_n = 64         #每一层的通道数（特征个数）
        self.depth = 7           #金字塔每一层深度学习的深度
        self.output_channel = 1  #重建层输出的通道数
        self.height = 320        #输入图像的高度
        self.width = 480        #输入输入的宽度
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
        self.model_name = "testModel04"  #训练模型的名字


        """
            新训练集部分
        """
        self.flicker = True    # 新训练集是否打开
        # self.flicker_adjust = True # 是否在处理图像时采用自动旋转和剔除,以满足需要的长宽,宽的最大值最好为240-320 之间

        self.mirflicker_dir = '/Users/wshwbluebird/ML/mirflickr/'  # mirflicker 训练集文件夹
        self.flicker_train_opt = 'xxx'  #mk训练集训练方式  random 为随机  如果不是random 就是按约定顺序
        self.flicker_file_index = 1  #mk训练集训练方式  连续型的训练指针
        self.flicker_begin_index = 1  # mk训练集训练方式  连续型的训练指针
        self.flicker_end_index = 25000  # mk训练集训练方式  连续型的训练指针

        self.batch_size = 16  # 每批训练数据的大小
        self.num_threads = 4  # 数据导入开启的线程数量
        self.min_after_dequeue = 1024  # 保证线程中至少剩下的数据数量
        self.flicker_random_list = list(np.arange(self.flicker_begin_index
                                                ,1+self.flicker_end_index,1))

    def predict(self,batchsize):
        self.batch_size = batchsize

    def get_image(self, path):
        im = Image.open(path)
        height = im.size[1]
        width = im.size[0]
        mark = False
        if width < height:
            mark = True
            height = width
        if height < self.height:
           return None,None,None

        if mark:
            im = im.rotate(90)

        box = [0,0,self.width,self.height]

        HR4 = im.crop(box)
        HR2 = HR4.resize((int(self.width /2),int(self.height / 2)),Image.BICUBIC)
        LR = HR4.resize((int(self.width / 4), int(self.height / 4)), Image.BICUBIC)
        return np.asarray(LR.convert('L'))\
            ,np.asarray(HR2.convert('L')), \
               np.asarray(HR4.convert('L'))



    def get_file_list(self):
        toFileName  =lambda x: join(self.mirflicker_dir,'im'+str(x)+'.jpg')
        if options.flicker_train_opt == 'random':
            numList =  random.sample(options.flicker_random_list, self.batch_size)
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

    """
        获得更清晰的flicker数据集
    """
    def get_pil_file_list(self):
        lr_list = []
        hr2_list = []
        hr4_list = []
        for i in range(10 *self.batch_size):
            filename = join(self.mirflicker_dir, 'im' + str(self.flicker_file_index) + '.jpg')
            lr,hr2,hr4 = self.get_image(filename)
            if lr is not None:
                lr_list.append(lr/255)
                hr2_list .append(hr2/255)
                hr4_list .append(hr4/255)
            self.flicker_file_index = self.flicker_begin_index - 1 \
                                      + (self.flicker_file_index + 1) % \
                                        (self.flicker_end_index - self.flicker_begin_index + 1)
            if len(lr_list) == self.batch_size:
                return lr_list,hr2_list,hr4_list



options = Argument()

if __name__ =='__main__':
     lr,ll,dd = options.get_pil_file_list()
     i = 0
     for image in ll:
         print(image)



# --utf8--#

class Argument():
    def __init__(self):
        self.input_channel = 3   #读取图片的通道数量
        self.conv_f = 3          #卷基层的边长
        self.conv_ft = 4         #反卷基层的边长
        self.conv_n = 64         #每一层的通道数（特征个数）
        self.depth = 6           #金字塔每一层深度学习的深度
        self.output_channel = 3  #重建层输出的通道数
        self.height = 320        #输入图像的高度
        self.width = 320         #输入输入的宽度
        self.batch_size = 16     #每批训练数据的大小
        self.num_threads = 4     #数据导入开启的线程数量
        self.min_after_dequeue = 1024  #保证线程中至少剩下的数据数量


options = Argument()

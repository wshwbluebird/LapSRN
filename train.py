
import data
import tensorflow as tf
import argument
import net
import time
import dacay_learning_rate
from os.path import join
import math
import os


def get_psnr_by_mse(mse):
    rmse = tf.sqrt(mse)
    ten = tf.constant(10.0)
    twenty = tf.constant(20.0)
    one = tf.constant(1.0)
    mid = tf.div(one,rmse)
    a = tf.log(mid)
    b = tf.log(ten)
    c = tf.div(a,b)
    psnr = tf.multiply(twenty, c)
    return psnr


def get_loss_of_batch(isFlicker):
    # [LR_set, HR2_set, HR4_set, HR8_set] = []
    """

    """

    """
        检查是否采用flicker数据集
    """
    if not isFlicker:
        [LR_set, HR2_set, HR4_set, HR8_set] = data.batch_queue_for_training_normal(data_path = argument.options.train_data_path)
    else:
        [LR_set, HR2_set, HR4_set, HR8_set] = data.pil_batch_queue()


    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(LR_set)
    loss1 = net.L1_Charbonnier_loss(hr2_predict, HR2_set)
    loss2 = net.L1_Charbonnier_loss(hr4_predict, HR4_set)
    loss_total = loss1 + loss2

    """
        检测是否加入8x层
    """
    if argument.options.max_log_scale == 3:
        loss3 = net.L1_Charbonnier_loss(hr8_predict, HR8_set)
        loss_total = loss1 + loss2 + loss3
    if argument.options.weight_decay!=0:
        loss_total += net.weight_decay_losses()

    return loss_total

"""
    获得每一张图像的监测信息
"""
def get_avg_psnr(path):
    [LR_set, HR2_set, HR4_set,_] = data.pil_single_test_SET5(path)
    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(LR_set)
    mse1 = tf.losses.mean_squared_error(HR2_set, hr2_predict)
    mse2 = tf.losses.mean_squared_error(HR4_set, hr4_predict)
    psnr1 = get_psnr_by_mse(mse1)
    psnr2 = get_psnr_by_mse(mse2)
    return [psnr1, psnr2]

def get_set5_psnr():
    list = data.get_all_file(argument.options.set5_dir, 'png')
    return  (get_avg_psnr(path=list[argument.options.test_index]))






def is_already_Save(savePath):
    return  os.path.exists(savePath+".meta")

"""
    训练用代码
"""
def train():

    save_path = join(argument.options.save_path,argument.options.model_name+".ckpt")
    path = tf.placeholder(tf.string)
    loss = get_loss_of_batch(argument.options.flicker)  #训练损失函数

    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = dacay_learning_rate.binary_decay(argument.options.lr, global_step, argument.options.decay_step
                                                , argument.options.decay)
    train_step  = tf.train.MomentumOptimizer(learning_rate,momentum=argument.options.momentum).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            if is_already_Save(save_path):
                saver.restore(sess, save_path)
                print("load last model ckpt")
            else:
                sess.run(tf.global_variables_initializer())
                print("create new model")
            tf.train.start_queue_runners(sess=sess)
            for step in range(1, argument.options.iter_nums+1):
                feed_dict = {path:argument.options.train_data_path}

                if step % 20 == 0:
                    feed_dict = {path:argument.options.train_data_path}

                start_time = time.time()
                step_,batch_loss = sess.run([train_step,loss],feed_dict=feed_dict)
                duration = time.time() - start_time

                if step % 20 ==0 :
                    b_loss_validation = sess.run(loss)
                    print("step " + str(step) + ", batch _loss in validation =" + str(b_loss_validation))

                elif step % 10 == 0 or step % 200 == 1:  # show training status
                    num_examples_per_step = argument.options.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = 'step %d, batch_loss_train = %.3f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (step, batch_loss, examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    save_path = saver.save(sess, save_path)
                    print("Model restored!"+str(sess.run(global_step)))


"""
    测试用代码

"""
def test():
    save_path = join(argument.options.save_path, argument.options.model_name + ".ckpt")
    if not is_already_Save(save_path):
        print("no model please train a model first")
        return
    psnr_all = get_set5_psnr()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        with tf.device('/cpu:0'):
            tf.train.start_queue_runners(sess=sess)
            psnr_s = sess.run(psnr_all)
            print(psnr_s)


# train()
test()



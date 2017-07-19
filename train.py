import data
import tensorflow as tf
import argument
import net
import time
from os.path import join
import os
def get_loss_of_batch(path):
    [LR_set, HR2_set, HR4_set, HR8_set] = data.batch_queue_for_training(argument.options.train_data_path)
    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(LR_set)
    loss1 = net.L1_Charbonnier_loss(hr2_predict, HR2_set)
    loss2 = net.L1_Charbonnier_loss(hr4_predict, HR4_set)
    loss3 = net.L1_Charbonnier_loss(hr8_predict, HR8_set)
    loss_total = loss1 + loss2 + loss3
    return loss_total




def is_already_Save(savePath):
    return  os.path.exists(savePath+".meta")

"""
    训练用代码
"""
def train():

    save_path = join(argument.options.save_path,argument.options.model_name+".ckpt")
    path = tf.placeholder(tf.string)
    loss = get_loss_of_batch(path)  #训练损失函数

    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.inverse_time_decay(argument.options.lr, global_step, argument.options.decay_step
                                                , argument.options.decay)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            if is_already_Save(save_path):
                saver.restore(sess, save_path)
                print("load last model ckpt")
            else:
                sess.run(tf.global_variables_initializer())
                print("create new model")
            tf.train.start_queue_runners(sess=sess)
            for step in range(1, argument.options.iter_nums+1):
                feed_dict = {path:argument.options.validation_data_path}

                if step % 200 == 0:
                    feed_dict = {path:argument.options.train_data_path}

                start_time = time.time()
                step_,batch_loss = sess.run([train_step,loss],feed_dict=feed_dict)
                duration = time.time() - start_time

                if step % 200 ==0 :
                    b_loss_validation = sess.run(loss)
                    print("step " + str(step) + ", batch _loss in validation =" + str(b_loss_validation))
                
                elif step % 100 == 0 or step % 200 == 1:  # show training status
                    num_examples_per_step = argument.options.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = 'step %d, batch_loss_train = %.3f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (step, batch_loss, examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    save_path = saver.save(sess, save_path)
                    print("Model restored!"+str(step))


"""
    测试用代码

"""
def test():
    save_path = join(argument.options.save_path, argument.options.model_name + ".ckpt")
    if not is_already_Save(save_path):
        print("no model please train a model first")
        return
    loss = get_loss_of_batch(argument.options.test_data_path)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            saver.restore(sess, save_path)
            tf.train.start_queue_runners(sess=sess)
            loss_total = 0
            for test_step in range(10):
                loss_cur = sess.run(loss)
                loss_total += loss_cur

            loss_result = loss_total/10
            print("loss in test_Set = " +str(loss_result))

"""
    预测用代码
"""
def inference():
    save_path = join(argument.options.save_path, argument.options.model_name + ".ckpt")
    if not is_already_Save(save_path):
        print("no model please train a model first")
        return
    pass

    #TODO


train()
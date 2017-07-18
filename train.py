import data
import tensorflow as tf
import argument
import build_network
import time
path = "./dataset/BSDS300/images/train"

input_height = argument.options.height
input_width = argument.options.width
input_channel =  argument.options.input_channel
input_batch = argument.options.batch_size

LR_set= tf.placeholder(tf.float32,[input_batch ,int(input_height/8),int(input_width/8),input_channel])
HR2_set= tf.placeholder(tf.float32,[input_batch ,int(input_height/4),int(input_width/4),input_channel])
HR4_set= tf.placeholder(tf.float32,[input_batch ,int(input_height/2),int(input_width/2),input_channel])
HR8_set= tf.placeholder(tf.float32,[input_batch ,input_height,input_width,input_channel])

hr2_predict, hr4_predict, hr8_predict = build_network.get_LasSRN(LR_set)

diff = tf.add(hr2_predict, -HR2_set)
eps = 1e-6
error = tf.sqrt(diff * diff + eps)
loss1 = tf.reduce_sum(error)

# loss1 = build_network.L1_Charbonnier_loss(hr2_predict, HR2_set)
# loss2 = build_network.L1_Charbonnier_loss(hr4_predict, HR4_set)
# loss3 = build_network.L1_Charbonnier_loss(hr8_predict, HR8_set)

print(hr2_predict, HR2_set)

# loss_total = loss1 + loss2 + loss3

global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.inverse_time_decay(argument.options.lr, global_step, argument.options.decay_step
                                            , argument.options.decay)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss1, global_step=global_step)
[lr,hr2,hr4,hr8] = data.batch_queue_for_training(argument.options.train_data_path)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    for step in range(1, argument.options.iter_nums + 1):
        start_time = time.time()
        [lr_set, hr2_set , hr4_set , hr8_set] = sess.run([lr,hr2,hr4,hr8])
        feed_dict = {LR_set: lr_set, HR2_set: hr2_set, HR4_set: hr4_set, HR8_set:hr8_set}
        batch_loss = sess.run(loss1, feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % 100 == 0:  # show training status
            num_examples_per_step = argument.options.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = 'step %d, batch_loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (step, batch_loss, examples_per_sec, sec_per_batch))
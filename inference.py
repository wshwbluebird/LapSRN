from os.path import join
import tensorflow as tf
import argument
import net
import os
import scipy.misc 

"""
    Helper functions
"""

def batch_queue_for_inference(data_paths):
    """
    patch all file paths in data_paths(list) to be ready to sent to tf
    """
    num_channel = argument.options.input_channel
    image_height = argument.options.height
    image_width = argument.options.width
    batch_size = argument.options.batch_size
    threads_num = argument.options.num_threads
    min_queue_examples = argument.options.min_after_dequeue

    filename_queue = tf.train.string_input_producer(data_paths)
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_jpeg(image_file, num_channel)
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    image_LR = tf.random_crop(patch, [image_height, image_width, num_channel])

    low_res_batch = tf.train.shuffle_batch(
        [image_LR],
        batch_size=batch_size,
        num_threads=threads_num,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return low_res_batch

def is_already_Save(savePath):
    """ check whether the model exists """
    return  os.path.exists(savePath+".meta")

def predict(data_paths):
    """
        tensorflow op, process all the input files in data_paths(list)
        return predicted 2x, 4x, 8x images
    """
    LR_set = batch_queue_for_inference(data_paths)
    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(LR_set)
    return [hr2_predict, hr4_predict, hr8_predict]


"""
   Prediction API
"""


def batch_inference(input_file_paths, output_dir_path, scale):
    """
    predict from given input_file_paths, return and store result in output_dir_path
    assume the scale is one from [2,4,8]
    """
    save_path = join(argument.options.save_path, argument.options.model_name + ".ckpt")
    if not is_already_Save(save_path):
        print("no model please train a model first")
        return
    hr2_predict, hr4_predict, hr8_predict = predict(input_file_paths)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        with tf.device('/gpu:0'):
            tf.train.start_queue_runners(sess=sess)
            for i in range(len(input_file_paths)):
                hr2, hr4, hr8 = sess.run([hr2_predict, hr4_predict, hr8_predict])
                if scale == 2:
                    hr_img = hr2.eval()
                elif scale == 4:
                    hr_img = hr4.eval()
                else:
                    hr_img = hr8.eval()

                scipy.misc.imsave(os.path.join(output_dir_path,"temp"+str(i)+".jpeg"), hr_img)

if __name__ == '__main__':
    """ a test """

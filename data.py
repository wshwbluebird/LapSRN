import tensorflow as tf
import argument
from os.path import join

def batch_queue_for_training(data_path):
    num_channel = argument.options.input_channel
    image_height = argument.options.height
    image_weight = argument.options.height
    batch_size = argument.options.batch_size
    threads_num = argument.options.num_threads
    min_queue_examples = argument.options.min_after_dequeue

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(join(data_path, '*.png')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, num_channel)
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    image_HR8 =  tf.random_crop(patch,[image_height,image_weight,num_channel])
    image_HR4 = tf.image.resize_images(image_HR8, [128, 128],  method=tf.image.ResizeMethod.BICUBIC)
    image_HR2 = tf.image.resize_images(image_HR4, [128, 128], method=tf.image.ResizeMethod.BICUBIC)
    image_LR  = tf.image.resize_images(image_HR2, [128, 128], method=tf.image.ResizeMethod.BICUBIC)

    low_res_batch, high2_res_batch,high4_res_batch,high8_res_batch = tf.train.shuffle_batch(
        [image_LR, image_HR2,image_HR4,image_HR8],
        batch_size=batch_size,
        num_threads=threads_num,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return low_res_batch, high2_res_batch, high4_res_batch, high8_res_batch
from os.path import join
import tensorflow as tf
import argument
import net
import os
import scipy.misc

path1 = "./dataset/BSDS300/images/train/"
def change_to_images(input_file_paths):
    """
    Args:通过路径列表获取 图片
        input_file_paths:

    Returns:

    """
    patch = []
    num_channel = argument.options.input_channel
    for path in input_file_paths:
        content = tf.read_file(path)
        image = None
        if path.endswith("jpg") or path.endswith("jpg"):
            image =  tf.image.decode_jpeg(content, num_channel)
        elif path.endswith("png"):
            image = tf.image.decode_png(content, num_channel)
        else :
            print("wrong image: "+path)

        if image is not None:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            patch.append(image)
        return patch


def save_image(image , path):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    if path.endswith("jpg") or path.endswith("jpg"):
        image = tf.image.encode_jpeg(image)
    elif path.endswith("png"):
        image = tf.image.encode_png(image)
    else:
        print("wrong image: " + path)
        return

    with open(path, "wb") as file:
        file.write(image)





def is_already_Save(savePath):
    """ check whether the model exists """
    print(savePath)
    print((savePath + ".meta"))
    print(os.getcwd())
    return os.path.exists(savePath + ".meta")


def predict(input_images):
    """
        tensorflow op, process all the input files in data_paths(input_images)
        return predicted 2x, 4x, 8x images
    """
    argument.options.predict(len(input_images))
    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(input_images)
    return [hr2_predict, hr4_predict, hr8_predict]


"""
   Prediction API
"""


def batch_inference(input_file_paths, output_dir_paths, scale_list):
    """

    Args:
        input_file_paths:  [list]
        output_dir_paths:  [list]  size same with input
        scale_list:   [list]    2,4,8 resume   size same with input

    Returns:

    """
    save_path = join(argument.options.save_path, argument.options.model_name + ".ckpt")
    if not is_already_Save(save_path):
        print("no model please train a model first")
        return

    input_images = change_to_images(input_file_paths)
    hr2_predict, hr4_predict, hr8_predict = predict(input_images)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        with tf.device('/cpu:0'):
            hr2, hr4, hr8 = sess.run([hr2_predict, hr4_predict, hr8_predict])
            hr_img= None
            for i in range(len(input_file_paths)):
                if scale_list[i] == 2:
                    hr_img = hr2[i]
                elif scale_list[i] == 4:
                    hr_img = hr4[i]
                else:
                    hr_img = hr8[i]

            save_image(hr_img,output_dir_paths[i])


if __name__ == '__main__':
    batch_inference(path1+'2092.jpg','./hr.jpg',[2])

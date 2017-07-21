from os.path import join
import tensorflow as tf
import argument
import net
import os
from PIL import Image
import scipy.misc

path1 = "./dataset/BSDS300/images/train/"
def change_to_image(path):
    """
    Args:通过路径列表获取 图片
        input_file_paths:

    Returns:

    """
    im = Image.open(path)
    height = im.size[0]
    weight = im.size[1]
    num_channel = argument.options.input_channel
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
        print(image)
        image = tf.reshape(image,(1,height,weight,3))
        print(image)
        return image





def save_image(image , path):
    with open(path, "wb") as file:
        file.write(image)





def is_already_Save(savePath):
    """ check whether the model exists """
    print(savePath)
    print((savePath + ".meta"))
    print(os.getcwd())
    return os.path.exists(savePath + ".meta")

def predict_single(input_image,path,scale):
    """
        tensorflow op, process all the input files in data_paths(input_image)
        return predicted 2x, 4x, 8x images
    """
    argument.options.predict(1)
    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(input_image)
    image = None
    if scale == 2:
        image = hr2_predict
    elif scale == 4:
        image = hr4_predict
    else:
        image = hr8_predict

    image = tf.image.convert_image_dtype(image[0], dtype=tf.uint8)
    if path.endswith("jpg") or path.endswith("jpg"):
        image = tf.image.encode_jpeg(image)
    elif path.endswith("png"):
        image = tf.image.encode_png(image)
    else:
        print("wrong image: " + path)
        return

    return image



"""
   Prediction API
"""




def single_inference(input_file_path, output_dir_path, scale):
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

    input_image = change_to_image(input_file_path)
    print(input_image)
    hr_predict= predict_single(input_image,input_file_path,scale)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        with tf.device('/cpu:0'):
            hr_img = sess.run(hr_predict)
            save_image(hr_img,output_dir_path)
            print('predict successfully in '+output_dir_path)

if __name__ == '__main__':
    single_inference(path1+'2092.jpg','./hr.jpg',2)

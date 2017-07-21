from os.path import join
import tensorflow as tf
import argument
import net
import os
from PIL import Image

path1 = "./dataset/BSDS300/images/train/"


def change_to_images(paths):
    """
    Args:通过路径列表获取 图片
        input_file_path: 输入图像的路径
    Returns:

    """
    im = Image.open(paths[0])
    height = im.size[1]
    width = im.size[0]
    pic_num = len(paths)
    num_channel = argument.options.input_channel
    images = []
    for path in paths:
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
            image = tf.reshape(image,(height,width,3))
            images.append(image)

    images = tf.reshape(images,(pic_num,height,width,3))
    return [images,height,width]


def save_image(image , path):
    with open(path, "wb") as file:
        file.write(image)


def get_scale_factor(scale):
    """
    Args:
        scale: ori scale

    Returns:
        floor scale
    """
    if scale <= 1:
        return 1
    elif scale <= 2:
        return 2
    elif scale <= 4:
        return 4
    else :
        return 8


def is_already_Save(savePath):
    """ check whether the model exists """
    return os.path.exists(savePath + ".meta")



"""
     单一图像的预测
"""


def predict_batch(input_image,paths,scale, ori_height, ori_width):
    """
        tensorflow op, process the input files in input_images
        return predicted 2x, 4x, 8x images
    """
    argument.options.predict(1)
    factor = get_scale_factor(scale)
    hr2_predict, hr4_predict, hr8_predict = net.get_LasSRN(input_image)
    hr_height = int(ori_height * scale)
    hr_width = int(ori_width * scale)


    """
        determine the floor picture
    """
    if factor == 1:
        images = input_image
    elif factor == 2:
        images = hr2_predict
    elif factor == 4:
        images = hr4_predict
    else:
        images = hr8_predict


    """
        resize to the real
    """
    images = [tf.image.resize_images(images[i], [hr_height, hr_width], method=tf.image.ResizeMethod.BICUBIC)
              for i in range(len(paths))]

    images = [ tf.image.convert_image_dtype(images[i], dtype=tf.uint8) for i in range(len(paths))]
    back_images = []
    for i in range(len(paths)):
        path = paths[i]
        image = images[i]
        if path.endswith("jpg") or path.endswith("jpg"):
            image = tf.image.encode_jpeg(image)
        elif path.endswith("png"):
            image = tf.image.encode_png(image)
        else:
            print("wrong image: " + path)
            return
        back_images.append(image)

    return back_images


"""
   Prediction API
"""


def single_inference(input_file_paths, output_dir_paths, scale):
    """
    Args:
        input_file_paths:  输入的路径path
        output_dir_paths:  输出的路径path
        scale_list:   [list]    1-8 resume   size same with input

    Returns:

    """
    save_path = join(argument.options.save_path, argument.options.model_name + ".ckpt")
    if not is_already_Save(save_path):
        print("no model please train a model first")
        return

    input_images,height,width = change_to_images(input_file_paths)
    hr_predict= predict_batch(input_images,input_file_paths,scale,height,width)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        with tf.device('/cpu:0'):
            for i in range(len(input_file_paths)):
                hr_img = sess.run(hr_predict)
                save_image(hr_img[i],output_dir_paths[i])
                print('predict successfully in '+output_dir_paths[i])

if __name__ == '__main__':
    single_inference([path1+'2092.jpg',path1+'8049.jpg'],['./hr1.jpg','./dd.jpg'],1.5)

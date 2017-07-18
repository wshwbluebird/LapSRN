# --utf8--#
import tensorflow as tf
import numpy as np
import layer
import argument as arg

# TODO
"""
    1 注释不完整需要补充
    2 option 参数还没有打包
    3 图像的resize 还没有解决 (可以抛到上一层解决,但是注释要写清楚)
"""


def upsample_filt(size):
    """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(channel, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    weights = np.zeros((channel,
                        channel,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(channel)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights


def featureExtraction(low_res_input):
    filters = arg.options.conv_f
    filters_tranpose = arg.options.conv_ft
    channel = arg.options.conv_n
    depth = arg.options.depth

    """
        特征提取层
    """
    with tf.name_scope("input"):
        """
            input layer
        """
        conv_input = layer.conv2d(low_res_input, filters, filters, channel, name='conv_input')
        lrelu_input = layer.leaky_relu(conv_input, leak=0.2, name='lrelu_input')

    with tf.name_scope("deep_cnn"):
        """
            cnn layers ,depth is given by options
        """
        last_lrelu = lrelu_input
        for i in range(depth):
            conv_cnn = layer.conv2d(last_lrelu, filters, filters, channel, isBias=False,
                                    name="block_conv_" + str(i + 1))
            lrelu_cnn = layer.leaky_relu(conv_cnn, 0.2, name="block_lrelu_" + str(i + 1))
            last_lrelu = lrelu_cnn

    with tf.name_scope("up_sampling"):
        """
            up_sampling layer
        """
        deconv_up = layer.deconv2d(last_lrelu, bilinear_upsample_weights(filters_tranpose, channel), [2, 2],
                                   name="up_sampling")
        up_samping_output = layer.leaky_relu(deconv_up, 0.2, name=" up_samping_output")

    return up_samping_output


def imageReconstruction(low_res_input, conv_up):
    filters_tranpose = arg.options.conv_ft
    channel = arg.options.output_channel
    filters = arg.options.conv_f

    """
        图像重构层
    """
    with tf.name_scope("Reconstruction"):
        """
            image reconstruction
        """
        deconv_image = layer.deconv2d(low_res_input, bilinear_upsample_weights(filters_tranpose, channel), [2, 2],
                                      name="deconv_image")
        conv_res = layer.conv2d(conv_up, filters, filters, channel, isBias=False, name="conv_res")

    HR = deconv_image + conv_res
    return HR


def get_LasSRN(low_res_input):
    """
    获得2x 4x 8x 三种预测结果
    Args:
        low_res_input:
        options:

    Returns:

    """
    convt_F1 = featureExtraction(low_res_input)
    HR_2 = imageReconstruction(low_res_input, convt_F1)

    convt_F2 = featureExtraction(convt_F1)
    HR_4 = imageReconstruction(HR_2, convt_F2)

    convt_F3 = featureExtraction(convt_F2)
    HR_8 = imageReconstruction(HR_4, convt_F3)

    return HR_2, HR_4, HR_8


def L1_Charbonnier_loss(predict, real):
    """
    损失函数
    Args:
        predict: 预测结果
        real:    真实结果

    Returns:
        损失代价

    """
    eps = 1e-6
    diff = tf.add(predict, -real)
    error = tf.sqrt(diff * diff + eps)
    loss = tf.reduce_sum(error)
    return loss

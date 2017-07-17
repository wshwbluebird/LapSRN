# --utf8--#
import tensorflow as tf


def conv2d(inputs, filter_height, filter_width, output_channels, stride=(1, 1), padding='SAME', isBias=True,
           name='Conv_2D', ):
    """
    tensorflow 的代码  用torch的风格进行封装
    Args:
        inputs:           上一层的输入
        filter_height:    卷积核的高
        filter_width:     卷积核的快
        output_channels:  输出通道的个数
        stride:           步长
        padding:          边界格式
        isBias:           是否需要偏置
        name:             这一层的名字 用于区分作用域
    Returns:

    """
    input_channels = int(inputs.get_shape()[-1])  # 通过上一层的输入获取 channelTrue
    fan_in = filter_height * filter_width * input_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, input_channels, output_channels]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        """
        有的话就话就reuse  没有的话就重新创建
        """
        filters = tf.get_variable(
            'weights', shape=weights_shape, initializer=filters_init, collections=['weights', 'variables'])
        if isBias:
            biases = tf.get_variable(
                'biases', shape=biases_shape, initializer=biases_init, collections=['biases', 'variables'])
            return tf.nn.conv2d(inputs, filters, strides=[1, *stride, 1], padding=padding) + biases
        else:
            return tf.nn.conv2d(inputs, filters, strides=[1, *stride, 1], padding=padding)


def deconv2d(inputs, filter_height, filter_width, output_shape, stride=(1, 1), padding='SAME', isBias = True , name='Deconv2D'):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = output_shape[-1]
    fan_in = filter_height * filter_width * output_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, output_channels, input_channels]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable(
            'weights', shape=weights_shape, initializer=filters_init, collections=['weights', 'variables'])

        if isBias:
            biases = tf.get_variable(
                'biases', shape=biases_shape, initializer=biases_init, collections=['biases', 'variables'])
            return tf.nn.conv2d_transpose(inputs, filters, output_shape, strides=[1, *stride, 1], padding=padding) + biases
        else:
            return tf.nn.conv2d_transpose(inputs, filters, output_shape, strides=[1, *stride, 1], padding=padding)

def relu(inputs, name='Relu'):
    return tf.nn.relu(inputs, name)


def leaky_relu(inputs, leak=0.1, name='LeakyRelu'):
    """
    Args:
        inputs: 上一层
        leak:    alpha
        name:    激活层的名字

    Returns:

    """
    with tf.name_scope(name):
        return tf.maximum(inputs, leak * inputs)


def batch_norm(inputs, decay, is_training, var_epsilon=1e-3, name='batch_norm'):
    """
    批量的标准化/归一化
    Args:
        inputs:
        decay:
        is_training:
        var_epsilon:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        scale = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]))
        offset = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]))
        avg_mean = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]), trainable=False)
        avg_var = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]), trainable=False)

        def get_batch_moments():
            batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(inputs.get_shape()) - 1)))
            assign_mean = tf.assign(avg_mean, decay * avg_mean + (1.0 - decay) * batch_mean)
            assign_var = tf.assign(avg_var, decay * avg_var + (1.0 - decay) * batch_var)
            with tf.control_dependencies([assign_mean, assign_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        def get_avg_moments():
            return avg_mean, avg_var

        mean, var = tf.cond(is_training, get_batch_moments, get_avg_moments)
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, var_epsilon)

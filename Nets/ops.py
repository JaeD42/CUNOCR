import tensorflow as tf


"""
Shape should be in form shape=[kernelsize,kernelsize,inFilter,outFilter]
"""
def conv_weight_bias(shape, stddev=0.1, c=0.1):
    W = tf.truncated_normal(shape, stddev=stddev)
    b = tf.constant(c, shape=[shape[-1]])

    return [tf.Variable(W), tf.Variable(b)]


def conv_2d(x, W, b=None, strides = [1,1,1,1], padding='SAME'):
    if b==None:
        return tf.nn.conv2d(x, W, strides=strides, padding=padding)
    else:
        temp = tf.nn.conv2d(x, W, strides=strides, padding=padding)
        return temp+b

"""
Copied from https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
"""
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def leaky_relu(x,leak=0.1):
    return tf.maximum(leak*x,x)
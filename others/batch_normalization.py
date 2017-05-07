import tensorflow as tf
import numpy as np


def batch_normalization(input, shape):
    mean, variance = tf.nn.moments(input, [0])
    offset = tf.Variable(tf.truncated_normal([shape], stddev=0.1), name='offset')
    scale = tf.Variable(tf.truncated_normal([shape], stddev=0.1), name='scale')
    return tf.nn.batch_normalization(input,
                                     mean,
                                     variance,
                                     offset,
                                     scale,
                                     variance_epsilon=1e-5)


x = tf.placeholder(tf.float32, [None, 784], name='x')
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

w = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=1.0), name='weight')
conv = tf.nn.conv2d(x_reshaped,
                    w,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
h = batch_normalization(conv, 32)
h_pool = tf.nn.max_pool(tf.nn.relu(h),
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_input = np.array([np.arange(784), np.arange(784)])

    h, h_pool = sess.run([h, h_pool], feed_dict={x: x_input})
    print(h[0].shape)
    print(h_pool[0].shape)

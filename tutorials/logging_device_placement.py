import tensorflow as tf
import numpy as np

with tf.device('/cpu:0'):
    a = tf.constant(np.arange(1.0, 7.0, 1), shape=[2, 3], name='a')
    b = tf.constant(np.arange(1.0, 7.0, 1), shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
config = tf.ConfigProto()
config.log_device_placement = True
with tf.Session(config=config) as sess:
    print(sess.run(c))

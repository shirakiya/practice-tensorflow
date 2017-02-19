import tensorflow as tf

x = tf.constant([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], name='x')
print(x)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(tf.rank(x)))
    print(sess.run(tf.shape(x)))

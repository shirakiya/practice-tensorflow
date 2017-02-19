import tensorflow as tf

W = tf.Variable([.3], tf.float32, name='weight')
b = tf.Variable([-.3], tf.float32, name='bias')
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

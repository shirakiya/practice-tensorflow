import tensorflow as tf

W = tf.Variable([.3], tf.float32, name='weight')
b = tf.Variable([-.3], tf.float32, name='bias')
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

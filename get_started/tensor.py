import tensorflow as tf

node1 = tf.constant(3.0, tf.float32, name='node1')
node2 = tf.constant(2.5, tf.float32, name='node2')
print(node1, node2)

node3 = tf.add(node1, node2, name='node3')
print(node3)

with tf.Session() as sess:
    print(sess.run([node1, node2]))
    print(sess.run(node3))

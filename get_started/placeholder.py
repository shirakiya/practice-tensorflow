import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
print(adder_node)

adder_and_triple = adder_node * 3
print(adder_and_triple)

with tf.Session() as sess:
    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

    print(sess.run(adder_and_triple, {a: 3, b: 4.5}))

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model(x, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2


def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    y = model(x, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(2000):
            x_batch, y_batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: x_batch,
                                                               y_: y_batch,
                                                               keep_prob: 1.0})
                print('step {step}, training accuracy {acc}'.format(step=i, acc=train_accuracy))
            sess.run(train_step, feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                      y_: mnist.test.labels,
                                                      keep_prob: 1.0})
        print('test accuracy {}'.format(test_accuracy))


def main(_):
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

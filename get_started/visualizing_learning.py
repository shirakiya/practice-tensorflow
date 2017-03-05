import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = weight_variable([input_dim, output_dim])
            variable_summaries(W)
        with tf.name_scope('bias'):
            b = bias_variable([output_dim])
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, W) + b
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    train_images = mnist.train.images
    tf.summary.image('train', train_images.reshape([55000, 28, 28, 1]), max_outputs=10)

    def feed_dict(is_train=True):
        if is_train:
            x_batch, y_batch = mnist.train.next_batch(50)
            k = 0.5
        else:
            x_batch = mnist.test.images
            y_batch = mnist.test.labels
            k = 1.0
        return {x: x_batch, y_: y_batch, keep_prob: k}

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    hidden1 = fc_layer(x, 784, 500, 'fc1')
    dropped = tf.nn.dropout(hidden1, keep_prob)
    y = fc_layer(dropped, 500, 10, 'fc2')

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)
        train_writer.flush()
        test_writer.flush()

        for i in range(1000):
            if i % 10 == 0:
                summary, train_accuracy = sess.run([merged, accuracy],
                                                   feed_dict=feed_dict(is_train=False))
                test_writer.add_summary(summary, i)
                print('step {step}, accuracy {acc}'.format(step=i, acc=train_accuracy))
            else:
                if i % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(is_train=True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
                    train_writer.add_summary(summary, i)
                else:
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(is_train=True))
                    train_writer.add_summary(summary, i)


def main(_):
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--summaries-dir', type=str, default='logs',
                        help='Directory for TensorBoard summaries')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

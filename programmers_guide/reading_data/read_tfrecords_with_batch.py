import os
import tensorflow as tf


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, value = reader.read(filename_queue)

    features = tf.parse_single_example(value, features={
        'file': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

    image = tf.image.decode_jpeg(features['image'], channels=3)
    resized_image = tf.image.resize_images(image, [480, 480],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.cast(features['label'], tf.int32)

    return resized_image, label


def generate_batch(image, label, min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 4

    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
        )
    else:
        image_batch, label_batch = tf.trian.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
        )

    return image_batch, label_batch


data_path = os.path.join(os.path.dirname(__file__), 'data')
tfrecord_path = os.path.join(data_path, 'images.tfrecords')

filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=None)
image, label = read_and_decode(filename_queue)

# (optional) written here for preprocessing if need.

batch_size = 2
num_examples_per_epoch = 5
min_fraction_of_examples_in_queue = 0.4
min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
image_batch, label_batch = generate_batch(image, label, min_queue_examples, batch_size)

init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # Coordinator helps multiple threads stop together and wait for them to stop
    coord = tf.train.Coordinator()
    # You haven't to use Coordinator to run start_queue_runners. Only want to user `coord.join()`
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for i in range(10):  # epoch = step // batch_size
            images, labels = sess.run([image_batch, label_batch])
            print('==========')
            print('labels: ', labels)
            print('labels shape: ', labels.shape)
            print('images: ', images.shape)
    except tf.errors.OutOfRangeError:
        print('exhaust datas from .tfrecords file.')
    finally:
        coord.request_stop()

    coord.join(threads)

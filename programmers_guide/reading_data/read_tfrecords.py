'''
ref.)
http://qiita.com/antimon2/items/c7d2285d34728557e81d
'''
import os
import tensorflow as tf

data_path = os.path.join(os.path.dirname(__file__), 'data')
tfrecord_path = os.path.join(data_path, 'images.tfrecords')

filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=1)

reader = tf.TFRecordReader()
filename, value = reader.read(filename_queue)  # hoge

features = tf.parse_single_example(value, features={
    'file': tf.FixedLenFeature([], tf.string),
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
})

file = tf.cast(features['file'], tf.string)
label = tf.cast(features['label'], tf.int32)
image = tf.image.decode_jpeg(features['image'], channels=3)

# this ops is nessesary to enable num_epochs in string_input_producer()
init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(init_op)

    try:
        while not coord.should_stop():
            image_file, image_tensor, label_val = sess.run([file, image, label])
            print('==========')
            print('file: ', image_file)
            print('label: ', label_val)
            print('image_tns: ', image_tensor)
            print('image_tns.shape: ', image_tensor.shape)
    except tf.errors.OutOfRangeError:
            print('Done session.')
    finally:
        coord.request_stop()

    coord.join(threads)

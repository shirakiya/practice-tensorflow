import os
import tensorflow as tf

data_path = os.path.join(os.path.dirname(__file__), 'data')
csv_path = os.path.join(data_path, 'files.csv')
image_path = os.path.join(data_path, 'images')

filename_queue = tf.train.string_input_producer([csv_path])

reader = tf.TextLineReader()
filename, line_value = reader.read(filename_queue)  # => "files.csv", Line value

record_defaults = [['image_0.jpg'], [0]]
col1, col2 = tf.decode_csv(line_value, record_defaults=record_defaults)  # => 'image.jpg', label

# extract image tensor
image_raw = tf.read_file(tf.string_join([image_path, col1], separator='/'))
image = tf.image.decode_jpeg(image_raw)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        source_file, imagefile, label = sess.run([filename, col1, col2])
        image_tensor = sess.run(image)
        print('==========')
        print('source file: ', source_file)
        print('image: ', imagefile)
        print('label: ', label)
        print('image_tns: ', image_tensor)
        print('image_tns.shape: ', image_tensor.shape)

    coord.request_stop()
    coord.join(threads)

import os
import tensorflow as tf

images_glob = os.path.join(os.path.dirname(__file__), 'data/images/*.jpg')

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(images_glob),
    shuffle=True
)

reader = tf.WholeFileReader()
filename, image_raw = reader.read(filename_queue)

image = tf.image.decode_jpeg(image_raw)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        image_tensor = sess.run(image)
        print('==========')
        print('filename: ', sess.run(filename))
        print('image_tns: ', image_tensor)
        print('image_tns.shape: ', image_tensor.shape)

    coord.request_stop()
    coord.join(threads)

import os
import tensorflow as tf

data_path = os.path.join(os.path.dirname(__file__), 'data')
image_path = os.path.join(data_path, 'images')
record_path = os.path.join(data_path, 'images.tfrecords')

datas = [
    ['image_0.jpg', 0],
    ['image_1.jpg', 1],
    ['image_2.jpg', 0],
    ['image_3.jpg', 0],
    ['image_4.jpg', 0],
]

with tf.python_io.TFRecordWriter(record_path) as writer:
    for data in datas:
        filename = data[0]
        label = data[1]

        # -TFRecord file format-
        #
        # features {
        #     feature {
        #         key: "age"
        #         value { float_list {
        #             value: 29.0
        #         }}
        #     }
        #     feature {
        #         key: "movie"
        #         value { bytes_list {
        #             value: "The Shawshank Redemption"
        #             value: "Fight Club"
        #     }}
        # }
        with open(os.path.join(image_path, filename), 'rb') as image_f:
            image = image_f.read()
            example = tf.train.Example(features=tf.train.Features(feature={
                'file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
        writer.write(example.SerializeToString())

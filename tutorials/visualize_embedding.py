import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

flags = tf.app.flags
flags.DEFINE_integer('vocabulary_size', 50000, 'File path of text data')
flags.DEFINE_integer('embedding_size', 128, 'Dimention of embedding vector')
flags.DEFINE_string('checkpoint', 'model.ckpt', 'File name of checkpoint')
FLAGS = flags.FLAGS

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
checkpoint_path = os.path.join(log_dir, FLAGS.checkpoint)


def main(_):
    embeddings = tf.Variable(
        tf.zeros([FLAGS.vocabulary_size, FLAGS.embedding_size]), name='embeddings')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings.name
        embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')

        summary_writer = tf.summary.FileWriter(log_dir)
        projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
    tf.app.run(main=main)

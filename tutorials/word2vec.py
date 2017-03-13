import os
import time
import collections
import random
import math
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(__file__), 'data')
log_dir = os.path.join(os.path.dirname(__file__), 'logs')

data_index = 0

flags = tf.app.flags
flags.DEFINE_string('data', 'text8', 'File path of text data')
flags.DEFINE_integer('vocabulary_size', 50000, 'File path of text data')
flags.DEFINE_integer('num_steps', 100001, 'Training step count.')
flags.DEFINE_string('checkpoint', 'model.ckpt', 'File name of checkpoint')
FLAGS = flags.FLAGS


def build_dataset(path, vocabulary_size):
    with open(os.path.join(data_path, path), 'r') as f:
        words = tf.compat.as_str(f.read()).split()
    print('Data size', len(words))

    c = collections.Counter(words)
    print('Original vocabulary size', len(c.keys()))

    count = [['UNK', -1]]
    count.extend(c.most_common(vocabulary_size - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    del words  # Hint to reduce memory
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def save_tsv(dictionary):
    with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        for word, index in dictionary.items():
            f.write('{0}\t{1}\n'.format(word, index))


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # = [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


def main(_):
    start = time.time()
    data, count, dictionary, reverse_dictionary = build_dataset(FLAGS.data, FLAGS.vocabulary_size)
    print('Most common words (+unknown)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    save_tsv(dictionary)

    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64    # Number of negative examples to sample.
    print('valid_examples: ', valid_examples)

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                tf.random_uniform([FLAGS.vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([FLAGS.vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)),
                name='nce_weights')
            nce_biases = tf.Variable(tf.zeros([FLAGS.vocabulary_size]), name='nce_biases')

        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=FLAGS.vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
    checkpoint_path = os.path.join(log_dir, FLAGS.checkpoint)

    with tf.Session(graph=graph) as sess:
        init.run()
        print('Initialized.')

        average_loss = 0
        for step in range(FLAGS.num_steps):
            batch_inputs, batch_lables = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_lables}

            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('step: {}\taverage_loss: {}'.format(step, average_loss))
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (- sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to {}:'.format(valid_word)
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '{} {},'.format(log_str, close_word)
                    print(log_str)
                saver.save(sess, checkpoint_path)
        final_embeddings = normalized_embeddings.eval()

    try:
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)
    except ImportError:
        print('Please install sklearn, matplotlib, and scipy to visualize embeddings.')

    print('time: {:.4f}'.format(time.time() - start))


if __name__ == '__main__':
    tf.app.run(main=main)

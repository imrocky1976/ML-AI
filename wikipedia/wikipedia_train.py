import tensorflow as tf
from wikipedia_model import EmbeddingModel
import wikipedia_data
from skipgrams import skipgrams
from batched import batched
import collections
import numpy as np


class AttrDict(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value


params = AttrDict(
    vocabulary_size = 10000,
    max_context = 10,
    embedding_size = 200,
    contrastive_examples = 100,
    learning_rate = 0.5,
    momentum = 0.5,
    batch_size = 1000,
)

TRAIN_LOG_DIR = 'logs'

data = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None])
model = EmbeddingModel(data, target, params)
corpus = wikipedia_data.Wikipedia(
    'https://dumps.wikimedia.org/enwiki/20180320/'
    'enwiki-20180320-pages-meta-current1.xml-p10p30303.bz2',
    wikipedia_data.WIKI_DOWNLOAD_DIR,
    params.vocabulary_size)
examples = skipgrams(corpus, params.max_context)
batches = batched(examples, params.batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()
# Get the summary writer.
summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)

average = collections.deque(maxlen=100)
for index, batch in enumerate(batches):
    feed_dict = {data: batch[0], target: batch[1]}
    cost, _ = sess.run([model.cost, model.optimize], feed_dict)
    average.append(cost)
    print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))
    if index % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, index)
        summary_writer.flush()
    if index > 100000:
        break

embeddings = sess.run(model.embeddings)
np.save(wikipedia_data.WIKI_DOWNLOAD_DIR + '/embeddings.npy', embeddings)
summary_writer.close()
sess.close()
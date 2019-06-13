import tensorflow as tf


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property


class EmbeddingModel:

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.embeddings
        self.cost
        self.optimize

    @lazy_property
    def embeddings(self):
        initial = tf.random_uniform([self.params.vocabulary_size,
                                     self.params.embedding_size],
                                    -1.0, 1.0) # shape=[10000,200]
        return tf.Variable(initial)

    @lazy_property
    def optimize(self):
        optimizer = tf.train.MomentumOptimizer(
            self.params.learning_rate, self.params.momentum)
        return optimizer.minimize(self.cost)

    @lazy_property
    def cost(self):
        embedded = tf.nn.embedding_lookup(self.embeddings, self.data)
        weight = tf.Variable(tf.truncated_normal([self.params.vocabulary_size,
                                                  self.params.embedding_size],
                                                 stddev=1.0 / self.params.embedding_size ** 0.5))
        bias = tf.Variable(tf.zeros([self.params.vocabulary_size]))
        target = tf.expand_dims(self.target, 1)
        loss = tf.reduce_mean(tf.nn.nce_loss(weight, bias, target, embedded,
            self.params.contrastive_examples, self.params.vocabulary_size))
        tf.summary.scalar('loss', loss)
        return loss

"""
Let's look at the relative code in word2vec example(examples/tutorials/word2vec).

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
This two lines create embedding representations. embeddings is a matrix where each row represents a word vector. embedding_lookup is a quick way to get vectors corresponding to train_inputs. In word2vec example, train_inputs consists of some int32 number, representing the id of target words. Basically, it can be placed by hidden layer feature.

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
These two lines create parameters. They will be updated by optimizer during training. We can use tf.matmul(embed, tf.transpose(nce_weights)) + nce_biases to get final output score. In other words, last inner-product layer in classification can be replaced by it.

loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,     # [vocab_size, embed_size]
                   biases=nce_biases,         # [embed_size]
                   labels=train_labels,       # [bs, 1]
                   inputs=embed,              # [bs, embed_size]
                   num_sampled=num_sampled, 
                   num_classes=vocabulary_size))
These lines create nce loss, @garej has given a very good explanation. num_sampled refers to the number of negative sampling in nce algorithm.

To illustrate the usage of nce, we can apply it in mnist example(examples/tutorials/mnist/mnist_deep.py) with following 2 steps:

Replace embed with hidden layer output. The dimension of hidden layer is 1024 and num_output is 10. Minimum value of num_sampled is 1. Remember to remove the last inner-product layer in deepnn().

y_conv, keep_prob = deepnn(x)                                            

num_sampled = 1                                                          
vocabulary_size = 10                                                     
embedding_size = 1024                                                    
with tf.device('/cpu:0'):                                                
  embed = y_conv                                                         
  # Construct the variables for the NCE loss                             
  nce_weights = tf.Variable(                                             
      tf.truncated_normal([vocabulary_size, embedding_size],             
                          stddev=1.0 / math.sqrt(embedding_size)))       
  nce_biases = tf.Variable(tf.zeros([vocabulary_size])) 
Create loss and compute output. After computing the output, we can use it to calculate accuracy. Note that the label here is not one-hot vector as used in softmax. Labels are the original label of training samples.

loss = tf.reduce_mean(                                   
    tf.nn.nce_loss(weights=nce_weights,                           
                   biases=nce_biases,                             
                   labels=y_idx,                                  
                   inputs=embed,                                  
                   num_sampled=num_sampled,                       
                   num_classes=vocabulary_size))                  

output = tf.matmul(y_conv, tf.transpose(nce_weights)) + nce_biases
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
When we set num_sampled=1, the val accuracy will end at around 98.8%. And if we set num_sampled=9, we can get almost the same val accuracy as trained by softmax. But note that nce is different from softmax.

Full code of training mnist by nce can be found here. Hope it is helpful.
"""
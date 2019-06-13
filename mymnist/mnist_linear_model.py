"""A MNIST linear network model contains inference, train, loss and evaluation."""

import tensorflow as tf
import math

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 32


def inference(images):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal(
            [IMAGE_PIXELS, HIDDEN1_UNITS], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN1_UNITS]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN1_UNITS, HIDDEN2_UNITS], stddev=1.0 / math.sqrt(float(HIDDEN1_UNITS))), name='weights')
        biases = tf.Variable(tf.zeros([HIDDEN2_UNITS]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Linear
    with tf.name_scope("softmax_linear"):
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN2_UNITS, NUM_CLASSES], stddev=1.0 / math.sqrt(float(HIDDEN2_UNITS))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss(logits, labels):
    labels = tf.to_int32(labels)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    return tf.reduce_mean(xentropy, name='xentropy_mean')


def train(total_loss, learning_rate=0.01):
    tf.summary.scalar('total_loss', total_loss)
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct_num = tf.reduce_sum(tf.to_float(correct))
    return correct_num / tf.cast(tf.shape(labels)[0], tf.float32)
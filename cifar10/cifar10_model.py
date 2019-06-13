"""Build the cifar10 CNN model."""

import tensorflow as tf

K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer output depth
N = 200  # fully connected layer

BATCH_SIZE = 100  # batch size for per-training
NUM_CLASS = 10  # data class number

IMAGE_BATCH_DEPTH = 3
IMAGE_BATCH_HEIGHT = 24
IMAGE_BATCH_WIDTH = 24


def random_distort_images(images):
    distort_images = None
    for i, image in enumerate(images):
        #print('random_distort_images %d/%d' % (i, BATCH_SIZE))
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(
            image, [IMAGE_BATCH_HEIGHT, IMAGE_BATCH_WIDTH, IMAGE_BATCH_DEPTH])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        distorted_image = tf.image.per_image_standardization(distorted_image)

        distorted_image = tf.expand_dims(distorted_image, 0)

        if distort_images is not None:
            distort_images = tf.concat([distort_images, distorted_image], 0)
        else:
            distort_images = distorted_image
    # Display the training images in the visualizer.
    tf.summary.image('images', distort_images)

    return distort_images

def batch_norm(logits, offset, iteration, is_test, convolutional=False):
    """
    Batch normalization. Use exponential moving average of mean and variance if
    evaluating testing data.
    :param logits: Input `Tensor` to be normalized
    :param offset: The offset `Tensor` to be added to the normalized tensor
    :param iteration: A scalar of the training step, this should be None if testing
    :param is_test: A scalar showing whether it is testing or training
    :param convolutional: A scalar showing whether it is a convolutional operation
    :return: bn_logits: Normalized logits
             update_moving_averages: An op of exponential moving average
    """
    if iteration is not None:
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    else:
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999)
    if convolutional:
        mean, variance = tf.nn.moments(logits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(logits, [0])
    bnepsilon = 1e-5
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    bn_logits = tf.nn.batch_normalization(logits, m, v, offset, None, bnepsilon)
    return bn_logits, update_moving_averages


def inference(images, iteration, is_test, keep_prob):
    """
    The layers of cifar10 CNN model:
    L0: normalize the input images [100, 24, 24, 3]
    L1: convolution [100, 24, 24, K]
        normalization [100, 24, 24, K]
        relu [100, 24, 24, K]
    L2: convolution [100, 24, 24, L]
        normalization [100, 24, 24, L]
        relu [100, 24, 24, L]
        max pool [100, 12, 12, L]
    L3: convolution [100, 12, 12, M]
        normalization [100, 12, 12, M]
        relu [100, 12, 12, M]
        max pool [100, 6, 6, M]
    L4: reshape [100, 6 * 6 * M]
        fully connection [100, N]
        normalization [100, N]
        relu [100, N]
    L5: dropout [100, N]
    L6: fully connection [100, NUM_CLASS]

    :param images: A tensor of the input images with shape (N_examples, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH).
    :param iteration: A scalar of the training step, this should be None if testing
    :param is_test: A scalar showing whether it is testing or training
    :param keep_prob: A scalar for the probability of dropout
    :returns: logits: A tensor of shape (N_examples, NUM_CLASS), with values
      equal to the logits of classifying the digit into one of 10 classes (the digits 0-9).
      update_ema: A group of tensorflow ops which updates the moving averages of batch normalizations.
    """
    with tf.name_scope('L0'):
        y0, update_ema0 = batch_norm(images, None, iteration, is_test, convolutional=True)

    with tf.name_scope('L1'):
        conv1 = tf.layers.conv2d(y0, filters=K, kernel_size=[6, 6], strides=[1, 1], padding='same',
                                 activation=None, bias_initializer=None)
        b1 = tf.Variable(tf.constant(0.1, shape=[K]))
        norm1, update_ema1 = batch_norm(conv1, b1, iteration, is_test, convolutional=True)
        y1 = tf.nn.relu(norm1)

    with tf.name_scope('L2'):
        conv2 = tf.layers.conv2d(y1, filters=L, kernel_size=[5, 5], strides=[1, 1], padding='same',
                                 activation=None, bias_initializer=None)
        b2 = tf.Variable(tf.constant(0.1, shape=[L]))
        norm2, update_ema2 = batch_norm(conv2, b2, iteration, is_test, convolutional=True)
        y2 = tf.nn.relu(norm2)
        pool2 = tf.layers.max_pooling2d(y2, 2, [2, 2], 'same')

    with tf.name_scope('L3'):
        conv3 = tf.layers.conv2d(pool2, filters=M, kernel_size=[4, 4], strides=[1, 1], padding='same',
                                 activation=None, bias_initializer=None)
        b3 = tf.Variable(tf.constant(0.1, shape=[M]))
        norm3, update_ema3 = batch_norm(conv3, b3, iteration, is_test, convolutional=True)
        y3 = tf.nn.relu(norm3)
        pool3 = tf.layers.max_pooling2d(y3, 2, [2, 2], 'same')

    with tf.name_scope('L4'):
        reshaped4 = tf.reshape(pool3, [-1, 6 * 6 * M])
        w4 = tf.Variable(tf.truncated_normal([6 * 6 * M, N], mean=0, stddev=0.1, dtype=tf.float32))
        b4 = tf.Variable(tf.constant(0.1, shape=[N]))
        fully_c4 = tf.matmul(reshaped4, w4)
        norm4, update_ema4 = batch_norm(fully_c4, b4, iteration, is_test, convolutional=False)
        y4 = tf.nn.relu(norm4)

    with tf.name_scope('L5'):
        y5 = tf.nn.dropout(y4, keep_prob)

    with tf.name_scope('L6'):
        w6 = tf.Variable(tf.truncated_normal([N, NUM_CLASS]))
        b6 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASS]))
        y6 = tf.matmul(y5, w6) + b6

    update_ema = tf.group(update_ema0, update_ema1, update_ema2, update_ema3, update_ema4)
    return y6, update_ema


def loss(logits, labels):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    return tf.reduce_mean(xentropy, name='xentropy_mean') * 100


def train(total_loss, learning_rate=0.001):
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    return tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    #return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
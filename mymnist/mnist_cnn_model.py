"""A MNIST CNN network model contains inference, train, loss and evaluation."""

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer


def batch_norm(logits, is_test, iteration, offset, convolutional=False):
    if iteration is not None:
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    else:
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(logits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(logits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda : exp_moving_avg.average(mean), lambda : mean)
    v = tf.cond(is_test, lambda : exp_moving_avg.average(variance), lambda : variance)
    logits_bn = tf.nn.batch_normalization(logits, m, v, offset, None, bnepsilon)
    return logits_bn, update_moving_averages


def inference(images, keep_prob, batchnorm=False, is_test=False, iteration=None):
    """Builds the graph for a deep net for classifying digits.

    Args:
      images: An input tensor with the dimensions (N_examples, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL).
      keep_prob: A scalar placeholder for the probability of dropout.

    Returns:
      A tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the digits 0-9).
    """
    # First convolutional layer - maps one grayscale image to 6 feature maps.
    with tf.name_scope('batchnorm0'):
        y0, update_ema0 = batch_norm(images, is_test, iteration, None, True)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([6, 6, 1, K])
        b_conv1 = bias_variable([K])
        h_conv1 = conv2d(y0, W_conv1)
        if batchnorm:
            y1_bn, update_ema1 = batch_norm(h_conv1, is_test, iteration, b_conv1, True)
            y1 = tf.nn.relu(y1_bn)
        else:
            y1 = tf.nn.relu(h_conv1 + b_conv1)  # [-1, 28, 28, K]

    # Pooling layer - downsamples by 2X.
    #with tf.name_scope('pool1'):
    #    h_pool1 = max_pool_2x2(h_conv1)  # [-1, 14, 14, K]

    # Second convolutional layer -- maps 6 feature maps to 12.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, K, L])
        b_conv2 = bias_variable([L])
        h_conv2 = conv2d(y1, W_conv2) #tf.nn.conv2d(y1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
        if batchnorm:
            y2_bn, update_ema2 = batch_norm(h_conv2, is_test, iteration, b_conv2, True)
            y2 = tf.nn.relu(y2_bn)
        else:
            y2 = tf.nn.relu(h_conv2 + b_conv2)  # [-1, 28, 28, L]

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(y2)  # [-1, 14, 14, L]

    # Third convolutional layer -- maps 12 feature maps to 24.
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([4, 4, L, M])
        b_conv3 = bias_variable([M])
        h_conv3 = conv2d(h_pool2, W_conv3) #tf.nn.conv2d(y2, W_conv3, strides=[1, 2, 2, 1], padding='SAME')
        if batchnorm:
            y3_bn, update_ema3 = batch_norm(h_conv3, is_test, iteration, b_conv3, True)
            y3 = tf.nn.relu(y3_bn)
        else:
            y3 = tf.nn.relu(h_conv3 + b_conv3)  # [-1, 14, 14, M]

    # Third pooling layer.
    with tf.name_scope('pool2'):
        h_pool3 = max_pool_2x2(y3)  # [-1, 7, 7, M]

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * M, N])
        b_fc1 = bias_variable([N])
        h_conv3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * M])
        h_fc1 = tf.matmul(h_conv3_flat, W_fc1)
        if batchnorm:
            y4_bn, update_ema4 = batch_norm(h_fc1, is_test, iteration, b_fc1, False)
            y4 = tf.nn.relu(y4_bn)
        else:
            y4 = tf.nn.relu(h_fc1 + b_fc1)  # [-1, N]

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(y4, keep_prob)

    # Map the 200 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([N, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # [-1, NUM_CLASSES]
    if batchnorm:
        update_ema = tf.group(update_ema0, update_ema1, update_ema2, update_ema3, update_ema4)
    else:
        update_ema = tf.no_op()
    return logits, update_ema


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def loss(logits, labels):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    return tf.reduce_mean(xentropy, name='xentropy_mean') * 100


def train(total_loss, learning_rate=0.001):
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    return tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    #return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

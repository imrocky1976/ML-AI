import tensorflow as tf
from cifar10_data import CLASS_NUM
from cifar10_data import MAX_TRAIN_STEPS

K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer output depth
N = 200  # fully connected layer


def inference(images, keep_prob, training):
    """
    The layers of cifar10 CNN inference model:
    input images [batch_size, 24, 24, 3]
    L1: convolution [batch_size, 24, 24, K]
        normalization [batch_size, 24, 24, K]
        relu [batch_size, 24, 24, K]
    L2: convolution [batch_size, 24, 24, L]
        normalization [batch_size, 24, 24, L]
        relu [batch_size, 24, 24, L]
        max pool [batch_size, 12, 12, L]
    L3: convolution [batch_size, 12, 12, M]
        normalization [batch_size, 12, 12, M]
        relu [batch_size, 12, 12, M]
        max pool [batch_size, 6, 6, M]
    L4: reshape [batch_size, 6 * 6 * M]
        fully connection [batch_size, N]
        normalization [batch_size, N]
        relu [batch_size, N]
    L5: dropout [batch_size, N]
    L6: fully connection [batch_size, NUM_CLASS]
    """

    with tf.name_scope('L1'):
        y1_conv = tf.layers.conv2d(images, filters=K, kernel_size=[6, 6],
                              strides=[1, 1], padding='same', activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                              bias_initializer=tf.constant_initializer(0.1))
        y1 = tf.layers.batch_normalization(y1_conv, training=training)

    with tf.name_scope('L2'):
        y2_conv = tf.layers.conv2d(y1, filters=L, kernel_size=[5, 5],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1))
        y2 = tf.layers.batch_normalization(y2_conv, training=training)
        y2 = tf.layers.max_pooling2d(y2, [2, 2], [2, 2], 'same')

    with tf.name_scope('L3'):
        y3_conv = tf.layers.conv2d(y2, filters=M, kernel_size=[4, 4],
                                   strides=[1, 1], padding='same', activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1))
        y3 = tf.layers.batch_normalization(y3_conv, training=training)
        y3 = tf.layers.max_pooling2d(y3, [2, 2], [2, 2], 'same')

    with tf.name_scope('L4'):
        y4 = tf.reshape(y3, [tf.shape(images)[0], 6 * 6 * M])
        w4 = tf.Variable(tf.truncated_normal([6 * 6 * M, N], stddev=0.1))
        b4 = tf.Variable(tf.constant(0.1, shape=[N]))
        y4 = tf.matmul(y4, w4) + b4
        y4 = tf.layers.batch_normalization(y4, training=training)
        y4 = tf.nn.relu(y4)

    with tf.name_scope('L5'):
        y5 = tf.nn.dropout(y4, keep_prob=keep_prob)

    with tf.name_scope('L6'):
        w6 = tf.Variable(tf.truncated_normal([N, CLASS_NUM], stddev=0.1))
        b6 = tf.Variable(tf.constant(0.1, shape=[CLASS_NUM]))
        y6 = tf.matmul(y5, w6) + b6

    return y6


def loss(logits, labels):
    """
    Calc total loss
    :param logits:
    :param labels:
    :return: total loss
    """

    with tf.name_scope('Loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='xentropy')
        total_loss = tf.reduce_mean(xentropy, name='xentropy_mean')

    return total_loss


def train(total_loss, global_step):
    """
    Train step
    :param total_loss:
    :return:
    """

    with tf.name_scope('Train_step'):
        # learning rate decay
        # max_learning_rate = 0.02
        # min_learning_rate = 0.0001
        # decay_speed = 1600
        # lr = min_learning_rate + (max_learning_rate - min_learning_rate) * \
        #                math.exp(-step / decay_speed)

        # decayed_learning_rate = learning_rate *
        # decay_rate ^ (global_step / decay_steps)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(0.1,
                                        global_step,
                                        MAX_TRAIN_STEPS / 3,
                                        0.9,
                                        staircase=True)

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', lr)
        # update variables of tf.layers.batch_normalization()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_step = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)
            # train_step = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step=global_step)
        	opt = tf.train.GradientDescentOptimizer(lr)
        	grads = opt.compute_gradients(total_loss)
        	# Apply gradients.
        	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

    return train_op


def eval(logits, labels):
    """
    Eval the accuracy of inference model
    :param logits:
    :param labels:
    :return: Accuracy of inference model
    """

    with tf.name_scope('Eval'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return accuracy
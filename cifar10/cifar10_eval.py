"""Functions for evaluating the cifar10 CNN model."""

import tensorflow as tf
from cifar10_input import Cifar10Data


def evaluation(logits, labels):
    """
    Evaluates image inference model.
    :param logits: A tensor of shape [BATCH_SIZE, NUM_CLASS], float32, each row represents an image inference
    :param labels: A tensor of shape [BATCH_SIZE], int32 or int64, the true id value of the image
    :return: Accuracy of image inference model.
    """
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def mass_evaluation(cifar10_data_obj, sess, eval_op, loss_op, images_pl, labels_pl, keep_prob_pl, is_test_pl):
    """
    Split massive testing images evaluation to 1000 images per-step due to the memory size limit of GPU
    :param cifar10_data_obj: Object of Cifar10Data.
    :param sess:
    :param eval_op:
    :param loss_op:
    :param images_pl:
    :param labels_pl:
    :param keep_prob_pl:
    :param is_test_pl:
    :return: Scalars of accuracy and loss
    """
    a = 0.0
    c = 0.0
    step = 0
    has_next = True
    cifar10_data_obj.init_testing_batch()
    while has_next:
        step = step + 1
        has_next, testing_image, testing_label = cifar10_data_obj.next_testing_batch(100)
        feed_dict = {images_pl: testing_image,
                     labels_pl: testing_label,
                     keep_prob_pl: 1.0,
                     is_test_pl: True}
        a_temp, c_temp = sess.run([eval_op, loss_op], feed_dict=feed_dict)
        a += a_temp
        c += c_temp
    a /= step
    c /= step
    return a, c

"""Trains and Evaluates the MNIST network using a feed dictionary."""

import tensorflow as tf
from six.moves import xrange
import time
import os
import sys
import mnist_cnn_model
import input_data
import math
import datavis
import numpy as np

MAX_STEPS = 10000
BATCH_SIZE = 100
LOG_DIR = './logs'
tf.set_random_seed(0)

def run_training():
    data_sets = input_data.read_data_sets('./input_data', reshape=False)

    # images_pl = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_model.IMAGE_PIXELS])
    # labels_pl = tf.placeholder(tf.int32, [BATCH_SIZE])
    images_pl = tf.placeholder(tf.float32, [None, mnist_cnn_model.IMAGE_SIZE, mnist_cnn_model.IMAGE_SIZE, 1])
    labels_pl = tf.placeholder(tf.int32)
    keep_prob_pl = tf.placeholder(tf.float32)
    learning_rate_pl = tf.placeholder(tf.float32)
    is_test_pl = tf.placeholder(tf.bool)
    iter_pl = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        logits, update_ema = mnist_cnn_model.inference(images_pl, keep_prob_pl, True, is_test_pl, iter_pl)

        total_loss = mnist_cnn_model.loss(logits, labels_pl)

        train_op = mnist_cnn_model.train(total_loss, learning_rate_pl)

        eval_op = mnist_cnn_model.evaluation(logits, labels_pl)

        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        sess.run(tf.global_variables_initializer())

        for step in xrange(MAX_STEPS):
            start_time = time.time()

            images_feed, labels_feed = input_data.random_batch(
                data_sets.training_image, data_sets.training_label, BATCH_SIZE
            )

            # learning rate decay
            max_learning_rate = 0.02 #0.003
            min_learning_rate = 0.0001
            decay_speed = 1600.0 #2000.0
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-step / decay_speed)

            feed_dict = {images_pl: images_feed,
                         labels_pl: labels_feed,
                         keep_prob_pl: 0.75,
                         learning_rate_pl: learning_rate,
                         is_test_pl: False,
                         iter_pl: step}
            sess.run(train_op, feed_dict=feed_dict)

            feed_dict = {images_pl: images_feed,
                         labels_pl: labels_feed,
                         keep_prob_pl: 1.0,
                         learning_rate_pl: learning_rate,
                         is_test_pl: False,
                         iter_pl: step}
            sess.run(update_ema, feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if (step + 1) % 100 == 0 or (step + 1) == MAX_STEPS:
                feed_dict = {images_pl: images_feed,
                             labels_pl: labels_feed,
                             keep_prob_pl: 1.0,
                             learning_rate_pl: learning_rate,
                             is_test_pl: False,
                             iter_pl: step}
                train_eval_val, loss_value = sess.run([eval_op, total_loss], feed_dict=feed_dict)
                print('Step %d: loss = %.2f, lr = %f (%.3f sec)' % (step, loss_value, learning_rate, duration))
                print('Training Data Eval: %.4f' % train_eval_val)

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                # feed_dict = {images_pl: data_sets.testing_image,
                #             labels_pl: data_sets.testing_label}
                # test_eval_val = sess.run(eval_op, feed_dict=feed_dict)
                test_eval_val, test_loss_val = eval_testing_data(
                    sess, eval_op, total_loss, images_pl, labels_pl, keep_prob_pl, is_test_pl,
                    data_sets.testing_image, data_sets.testing_label)
                print('Testing Data Eval: ' + str(test_eval_val) + '  loss: ' + str(test_loss_val))

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

        summary_writer.close()


def eval_testing_data(sess, eval_op, loss_op, images_pl, labels_pl, keep_prob_pl, is_test_pl, testing_image, testing_label):
    a = 0.0
    c = 0.0
    for step in range(testing_image.shape[0] // 1000):
        index_low = step * 1000
        index_high = testing_image.shape[0]
        if step != testing_image.shape[0] // 1000:
            index_high = (step + 1) * 1000
        feed_dict = {images_pl: testing_image[index_low:index_high],
                     labels_pl: testing_label[index_low:index_high],
                     keep_prob_pl: 1.0,
                     is_test_pl: True}
        a_temp, c_temp = sess.run([eval_op, loss_op], feed_dict=feed_dict)
        a += a_temp
        c += c_temp
    a /= (step + 1)
    c /= (step + 1)
    return a, c


def run_predicting():
    pred_images = input_data.read_image_data('./pred_data/image.data', reshape=False)
    pred_labels = input_data.read_label_data('./pred_data/label.data')

    num_examples = pred_images.shape[0]
    images_pl = tf.placeholder(tf.float32, [None, mnist_cnn_model.IMAGE_SIZE, mnist_cnn_model.IMAGE_SIZE, 1])
    labels_pl = tf.placeholder(tf.int32)
    #keep_prob_pl = tf.placeholder(tf.float32)
    #learning_rate_pl = tf.placeholder(tf.float32)
    is_test_pl = tf.placeholder(tf.bool)
    #iter_pl = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        logits, _ = mnist_cnn_model.inference(images_pl,
                                              keep_prob=1.0, batchnorm=True, is_test=is_test_pl, iteration=None)
        pred_op = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)  # predict result

        correct_op = tf.nn.in_top_k(logits, labels_pl, 1)  # True or False

        correct_sum_op = tf.reduce_sum(tf.cast(correct_op, tf.float32))

        ckpt = tf.train.get_checkpoint_state(LOG_DIR)

        saver = tf.train.Saver()

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            correct_sum = 0.0
            wrong_images = None
            wrong_output_labels = None
            for step in range(num_examples // 1000):
                print("step", str(step), '...')
                low = step * 1000
                high = (step + 1) * 1000
                if high > num_examples:
                    high = num_examples

                feed_dict = {images_pl: pred_images[low:high], labels_pl: pred_labels[low:high], is_test_pl: True}
                output_labels = sess.run(pred_op, feed_dict=feed_dict)
                correct = sess.run(correct_op, feed_dict=feed_dict)
                for i, val in enumerate(correct):
                    if not val:
                        temp_image = np.reshape(pred_images[low + i],
                                                      [1, mnist_cnn_model.IMAGE_SIZE, mnist_cnn_model.IMAGE_SIZE, 1])
                        temp_label = np.array(output_labels[i])
                        if wrong_images is None:
                            wrong_images = temp_image
                        else:
                            wrong_images = np.concatenate((wrong_images, temp_image))

                        if wrong_output_labels is None:
                            wrong_output_labels = temp_label
                        else:
                            wrong_output_labels = np.append(wrong_output_labels, temp_label)
                correct_sum += sess.run(correct_sum_op, feed_dict=feed_dict)
            print("accuracy: %f" % (correct_sum / num_examples))
            datavis.data_vis(wrong_images, wrong_output_labels)
        else:
            print('You must train before use!')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-t':
            run_training()
            sys.exit(0)
    run_predicting()


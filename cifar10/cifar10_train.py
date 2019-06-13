"""Trains the cifar10 network using a feed dictionary."""

import tensorflow as tf
from six.moves import xrange
import time
import os
import cifar10_eval
import cifar10_model
from cifar10_input import Cifar10Data
import math

MAX_STEPS = 10000
LOG_DIR = './logs'
tf.set_random_seed(0)


def run_training():
    cifar10_data = Cifar10Data('./input_data')

    images_pl = tf.placeholder(
        tf.float32, [None, cifar10_model.IMAGE_BATCH_HEIGHT, cifar10_model.IMAGE_BATCH_WIDTH, cifar10_model.IMAGE_BATCH_DEPTH])
    labels_pl = tf.placeholder(tf.int32)
    keep_prob_pl = tf.placeholder(tf.float32)
    learning_rate_pl = tf.placeholder(tf.float32)
    is_test_pl = tf.placeholder(tf.bool)
    iter_pl = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        logits, update_ema = cifar10_model.inference(images_pl, iter_pl, is_test_pl, keep_prob_pl)
        total_loss = cifar10_model.loss(logits, labels_pl)
        train_op = cifar10_model.train(total_loss, learning_rate_pl)
        eval_op = cifar10_eval.evaluation(logits, labels_pl)

        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())

        # learning rate decay
        max_learning_rate = 0.02  # 0.003
        min_learning_rate = 0.0001
        decay_speed = 1600.0  # 2000.0
        for step in xrange(MAX_STEPS):
            print('step %d/%d' % (step, MAX_STEPS))
            start_time = time.time()
            images_feed, labels_feed = cifar10_data.random_training_batch(cifar10_model.BATCH_SIZE)
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-step / decay_speed)

            images_feed = sess.run(cifar10_model.random_distort_images(images_feed))

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
                train_eval_val, loss_value = sess.run([eval_op, total_loss], feed_dict=feed_dict)
                print('Step %d: loss = %.2f, lr = %f (%.3f sec)' % (step + 1, loss_value, learning_rate, duration))
                print('Training Data Eval: %.4f' % train_eval_val)

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                # Evaluate the model periodically.
                # feed_dict = {images_pl: data_sets.testing_image,
                #             labels_pl: data_sets.testing_label}
                # test_eval_val = sess.run(eval_op, feed_dict=feed_dict)
                test_eval_val, test_loss_val = cifar10_eval.mass_evaluation(cifar10_data,
                    sess, eval_op, total_loss, images_pl, labels_pl, keep_prob_pl, is_test_pl)
                print('Testing Data Eval: ' + str(test_eval_val) + '  loss: ' + str(test_loss_val))

            # Save a checkpoint periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

        summary_writer.close()


if __name__ == '__main__':
    run_training()


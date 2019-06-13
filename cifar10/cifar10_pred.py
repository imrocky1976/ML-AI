"""Functions for predicting images using the CNN cifar10 model."""

import tensorflow as tf
import datavis
import numpy as np
import cifar10_train
from cifar10_input import Cifar10Data
import cifar10_model
import cifar10_eval


def prediction(logits):
    """
    Predicts images using the inference model.
    :param logits: A tensor of shape [BATCH_SIZE, NUM_CLASS], float32, each row represents an image inference
    :return: Returns a tensor of class ids of input logits
    """
    results = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)
    return results


def central_crop_images(images):
    crop_images = None
    for image in images:
        image = tf.image.central_crop(image, 24 / 32)
        image = tf.expand_dims(image, 0)

        if crop_images is not None:
            crop_images = tf.concat([crop_images, image], 0)
        else:
            crop_images = image
    # Display the training images in the visualizer.
    tf.summary.image('images', crop_images)

    return crop_images


def run_predicting(class_names, images, real_labels=None):
    """
    Run image predicting. Use matplotlib to draw the predicting results.
    :param class_names: names of labels.
    :param images: A numpy array of shape [NUM, HEIGHT, WIDTH, DEPTH], float32, represents the images to predict.
    :param real_labels: A numpy array of shape [NUM], int32, each element represents the class id of
    the image to predict.
    :return: if `pred_labels` is not none, returns the accuracy, else return none.
    """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # images = sess.run(central_crop_images(images))

        # logits, _ = cifar10_model.inference(tf.convert_to_tensor(images), iteration=None, is_test=tf.convert_to_tensor(True), keep_prob=1.0)
        logits, _ = cifar10_model.inference(central_crop_images(images), iteration=None,
                                            is_test=tf.convert_to_tensor(True), keep_prob=1.0)
        pred_op = prediction(logits)

        ckpt = tf.train.get_checkpoint_state(cifar10_train.LOG_DIR)
        saver = tf.train.Saver()

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            pred_labels = sess.run(pred_op)
            pred_labels = [class_names[i] for i in pred_labels]
            if real_labels is not None:
                eval_op = cifar10_eval.evaluation(logits, real_labels)
                accuracy = sess.run(eval_op)
                print("accuracy: %f" % accuracy)
            rlbl = [class_names[i] for i in real_labels]
            datavis.data_vis(images, pred_labels, rlbl)
        else:
            print('You must train before use!')



if __name__ == '__main__':
    cifar10_data = Cifar10Data('./input_data')
    run_predicting(cifar10_data.label_names,
                   cifar10_data.testing_images[0:10],
                   cifar10_data.testing_labels[0:10])
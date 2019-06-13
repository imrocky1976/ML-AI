# -*- coding:utf-8 -*-

import tensorflow as tf
import os

W = [[ 0.82579714], [-0.10430956], [-1.34950912], [ 2.63314915], [-0.01525408]]
b = -0.628012

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/titanic/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def inputs():
    """ 读取或生成训练数据X及其期望输出Y

    """
    passenger_id, survived, pclass, name, sex, age, sibsp, \
    parch, ticket, fare, cabin, embarked = read_csv(418, 'test.csv', [[0.], [0.],
                                                                       [0], [""], [""],
                                                                       [0.], [0.], [0.],
                                                                       [""], [0.], [""],
                                                                       [""]])
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    features = tf.transpose([is_first_class, is_second_class, is_third_class, gender, age])
    survived = tf.reshape(survived, [418, 1])

    return features, survived


def combine_inputs(X):
    #return tf.reduce_sum(tf.multiply(X, W), axis=1) + b
    return tf.matmul(X, W) + b

def inference(X):
    """ 计算推断模型在数据X上的输出，并将结果返回

    """
    return tf.sigmoid(combine_inputs(X))

def evaluate(sess, X, Y):
    """ 对训练得到的模型进行评估
    :param sess:
    :param X:
    :param Y:
    :return:
    """

    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))


if __name__ == "__main__":
    with tf.Session() as sess:
        X, Y = inputs()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        evaluate(sess, X, Y)

        coord.request_stop()
        coord.join(threads)


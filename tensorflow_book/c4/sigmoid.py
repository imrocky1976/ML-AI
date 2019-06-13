# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__)
                                                     + "/titanic/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)




#初始化变量和模型参数，定义训练闭环中的运算
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")

def combine_inputs(X):
    #return tf.reduce_sum(tf.multiply(X, W), axis=1) + b
    return tf.matmul(X, W) + b

def inference(X):
    """ 计算推断模型在数据X上的输出，并将结果返回

    """
    return tf.sigmoid(combine_inputs(X))

def loss(X, Y):
    """ 依据训练数据X及其期望输出Y计算损失

    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

def inputs():
    """ 读取或生成训练数据X及其期望输出Y

    """
    passenger_id, survived, pclass, name, sex, age, sibsp, \
    parch, ticket, fare, cabin, embarked = read_csv(891, 'train.csv', [[0.], [0.],
                                                                       [0], [""], [""],
                                                                       [0.], [0.], [0.],
                                                                       [""], [0.], [""],
                                                                       [""]])
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    features = tf.transpose([is_first_class, is_second_class, is_third_class,
                             gender, age])
    survived = tf.reshape(survived, [891, 1])

    return features, survived

def train(total_loss):
    """ 依据计算的总损失训练或调整模型参数
    :param total_loss:
    :return:
    """
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    """ 对训练得到的模型进行评估
    :param sess:
    :param X:
    :param Y:
    :return:
    """

    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))


#创建一个Saver对象
#saver = tf.train.Saver()

#在一个会话对象中启动数据流图，搭建流程
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(5):
        name = 'W%d' % i
        tf.summary.scalar(bytes(name), W[i][0])
    tf.summary.scalar(b'b', b)

    merged_summaries = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./c4/sigmoid-graph', graph=sess.graph)

    # 实际的训练迭代次数
    train_steps = 30000
    for step in range(train_steps):
        _, summary = sess.run([train_op, merged_summaries])
        writer.add_summary(summary, global_step=step)


    print sess.run(W)
    print sess.run(b)
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)


    writer.flush()
    writer.close()
    sess.close()
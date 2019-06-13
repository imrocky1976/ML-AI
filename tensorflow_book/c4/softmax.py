# -*- coding:utf-8 -*-

import tensorflow as tf
import os


def read_csv(batch_size, file_name, record_defaults):
    print("data dir:%s" % os.path.dirname(__file__) + "/iris/" + file_name)
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/iris/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)




#初始化变量和模型参数，定义训练闭环中的运算
feature_number = 4
class_number = 3

W = tf.Variable(tf.zeros([feature_number, class_number]), name="weights")
b = tf.Variable(tf.zeros([class_number]), name="bias")

def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    """ 计算推断模型在数据X上的输出，并将结果返回

    """
    return tf.nn.softmax(combine_inputs(X))

def loss(X, Y):
    """ 依据训练数据X及其期望输出Y计算损失

    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

def inputs():
    """ 读取或生成训练数据X及其期望输出Y

    """
    sepal_length, sepal_width, petal_length, petal_width, label = read_csv(150, 'iris.csv', [[0.], [0.], [0.], [0.], [""]])

    label_number = tf.to_int32(tf.argmax(tf.to_int32([tf.equal(label, ["Iris-setosa"]), tf.equal(label, ["Iris-versicolor"]),
                                                      tf.equal(label, ["Iris-virginica"])]), 0))

    features = tf.transpose([sepal_length, sepal_width, petal_length, petal_width])

    return features, label_number

def train(total_loss):
    """ 依据计算的总损失训练或调整模型参数
    :param total_loss:
    :return:
    """
    learning_rate = 0.1
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    """ 对训练得到的模型进行评估
    :param sess:
    :param X:
    :param Y:
    :return:
    """

    predicted = tf.cast(tf.argmax(inference(X), 1), tf.int32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))


#创建一个Saver对象
#saver = tf.train.Saver()
#在一个会话对象中启动数据流图，搭建流程

with tf.Session() as sess:
    X, Y = inputs()
    sess.run(tf.global_variables_initializer())

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(feature_number):
        for j in range(class_number):
            name = 'W%d%d' % (i, j)
            tf.summary.scalar(bytes(name), W[i][j])
    for i in range(class_number):
        name = 'b%d' % i
        tf.summary.scalar(bytes(name), b[i])

    tf.summary.scalar(b'loss', total_loss)

    merged_summaries = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./c4/softmax-graph', graph=sess.graph)

    # 实际的训练迭代次数
    train_steps = 3000
    for step in range(train_steps):
        _, summary = sess.run([train_op, merged_summaries])
        writer.add_summary(summary, global_step=step)
        print("step: %d" % step)


    print sess.run(W)
    print sess.run(b)
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)


    writer.flush()
    writer.close()
    sess.close()
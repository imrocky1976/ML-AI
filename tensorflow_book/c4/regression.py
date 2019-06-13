# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np

#初始化变量和模型参数，定义训练闭环中的运算
W = tf.Variable(tf.zeros([1]), name="weights") # W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")

def inference(X):
    """ 计算推断模型在数据X上的输出，并将结果返回

    """
    return tf.multiply(X, W) + b # tf.matmul(X, W) + b

def loss(X, Y):
    """ 依据训练数据X及其期望输出Y计算损失

    """
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

def inputs():
    """ 读取或生成训练数据X及其期望输出Y

    """
    """
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57],
                  [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60],
                  [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451,
                         302, 288, 385, 402, 365,
                         209, 290, 346, 254, 395,
                         434, 220, 374, 308, 220,
                         311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)
    """
    x = np.arange(0, 100, 0.2)
    xArr = []
    yArr = []
    for i in x:
        xArr.append(i)
        yArr.append(0.5 * i + 3 + np.random.uniform(0, 1) * 4 * np.math.sin(i))
    return tf.to_float(xArr), tf.to_float(yArr)
def train(total_loss):
    """ 依据计算的总损失训练或调整模型参数
    :param total_loss:
    :return:
    """
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    """ 对训练得到的模型进行评估
    :param sess:
    :param X:
    :param Y:
    :return:
    """
    print sess.run(inference([80., 25.])) # ~ 303
    print sess.run(inference([65., 25.])) # ~256


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

    #验证之前是否已经保存了检查点文件
#    initial_step = 0
#    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
#    if ckpt and ckpt.model_checkpoint_path:
        #从检查点恢复模型参数
#        saver.restore(sess, ckpt.model_checkpoint_path)
#        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    tf.summary.scalar(b'W', W[0])
    tf.summary.scalar(b'b', b)

    merged_summaries = tf.summary.merge_all()

    writer = tf.summary.FileWriter('my-graph', graph=sess.graph)

    # 实际的训练迭代次数
    train_steps = 300000
#    for step in range(initial_step, train_steps):
    for step in range(train_steps):
        _, summary = sess.run([train_op, merged_summaries])
        writer.add_summary(summary, global_step=step)

        #定期保存训练检查点
#        if step % 100 == 0:
#            saver.save(sess, 'my-model', global_step=step)
    print sess.run(W)
    print sess.run(b)
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join()

    # 保存训练检查点
#    saver.save(sess, 'my-model', global_step=train_steps)


    writer.flush()
    writer.close()
    sess.close()
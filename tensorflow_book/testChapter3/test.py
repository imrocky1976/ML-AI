# -*- coding:utf-8 -*-

import tensorflow as tf

#显式创建一个Graph对象
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("variables"):
        #追踪数据流图运行次数的Variable对象
        global_step = tf.Variable(0, dtype=tf.int32,
                                  trainable=False, name="global_step")

        #追踪所有输出随时间的累加和的Variable对象
        total_output = tf.Variable(0.0, dtype=tf.float32,
                                   trainable=False, name="total_output")


    #主要的变换Op
    with tf.name_scope("transformation"):

        #独立的输入层
        with tf.name_scope("input"):
            #创建可接收一个向量的占位符
            a = tf.placeholder(tf.float32, shape=[None],
                                name="input_placeholder_a")

        #独立的中间层
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")

        #独立的输出层
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")


    with tf.name_scope("update"):
        #用最新的输出更新Variable对象totaloutput
        update_total = total_output.assign_add(output)

        #将Variable对象globalstep增1，只要数据流图运行，该操作便需要进行
        increment_step = global_step.assign_add(1)


    #汇总Op
    with tf.name_scope("summaries"):
        avg = tf.div(update_total, tf.cast(increment_step,
                                            tf.float32), name="average")

        #为输出节点创建汇总数据
        tf.summary.scalar(b'Output', output) #, name="output_summary"

        tf.summary.scalar(b'Sum of outputs over time', update_total) #, name="total_summary"

        tf.summary.scalar(b'Average of outputs over time', avg) #, name="avg_summary"


    #全局Variable对象和Op
    with tf.name_scope("global_ops"):
        #初始化Op
        init = tf.global_variables_initializer()
        #将所有汇总数据合并到一个Op中
        merged_summaries = tf.summary.merge_all()



#用明确创建的Graph对象启动一个会话
sess = tf.Session(graph=graph)

#开启一个SummaryWriter对象，保存汇总数据
writer = tf.summary.FileWriter('./improved_graph', graph)

#初始化Variable对象
sess.run(init)



def run_graph(input_tensor):
    """
    辅助函数；用给定的输入张量运行数据流图，
    并保存汇总数据
    :param input_tensor:
    :return:
    """
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries],
                                feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)




#用不同输入运行该数据流图
run_graph([2,8])
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11,4])
run_graph([4,1])
run_graph([7,3,1])
run_graph([6,3])
run_graph([0,2])
run_graph([4,5,6])

#将汇总数据写入磁盘
writer.flush()

#关闭FileWriter对象
writer.close()

#关闭Session对象
sess.close()

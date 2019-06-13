#encoding:utf-8

import tensorflow as tf
import numpy as np

#2*2的零矩阵
zeros = tf.zeros([2, 2], dtype=tf.int32)

#长度为6的全1向量
ones = tf.ones([6])

#3*3的张量，元素服从0-10均匀分布
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)

#3*3*3的张量，元素服从0均值，标准差为2的正态分布
normal = tf.random_normal([3, 3, 3], mean=0, stddev=2)

#不会创建任何偏离均值超过2倍标准差的值；不会返回任何小于3或大于7的值
trunc = tf.truncated_normal([3, 3, 3], mean=5, stddev=1)

sess = tf.Session()

print sess.run(zeros)
print sess.run(ones)
print sess.run(uniform)
print sess.run(normal)
print sess.run(trunc)

var1 = tf.Variable([1, 3], dtype=tf.int32, name="var1", trainable=False)
random_var = tf.Variable(tf.random_normal([2, 2], mean=0, stddev=2))

init_var1 = tf.variables_initializer([var1], name="init_var1")
sess.run(init_var1)
print sess.run(var1)

var1_times_two = var1.assign(var1 * 2)
print sess.run(var1_times_two)
print sess.run(var1)

print sess.run(var1.assign_add([5, 5]))
print sess.run(var1.assign_sub([1, 1]))

#init = tf.global_variables_initializer()
#sess.run(init)
#print sess.run(random_var)

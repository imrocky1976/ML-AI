import numpy as np
import tensorflow as tf

a = tf.constant([5, 3], name = "input_a")
b = tf.reduce_prod(a, name = "prod_b")
c = tf.reduce_sum(a, name = "sum_c")
d = tf.add(b, c, name = "add_d")

sess = tf.Session()
print("output is %d" % sess.run(d))
writer = tf.summary.FileWriter("./my_graph", sess.graph)
writer.close()
sess.close()

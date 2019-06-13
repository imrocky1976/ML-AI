import numpy as np
import tensorflow as tf

g1 = tf.Graph()
g2 = tf.get_default_graph()

with g1.as_default():
#    a = tf.constant(np.array([3, 5], dtype = np.int32), name = "input_a")
#    b = tf.constant(np.array([4, 6], dtype = np.int32), name = "input_b")
    with tf.name_scope("Input"):
        a = tf.placeholder(np.int32, shape=[2], name="input_a")
        b = tf.placeholder(np.int32, shape=[2], name="input_b")
    with tf.name_scope("Transformation"):
        c = tf.add(a , b, name = "add_c")
        d = tf.multiply(a, b, name = "mult_d")
        e = tf.add(c, d, name = "add_e")

replace = {a:np.array([3,5], dtype=np.int32), b:np.array([4,6], dtype=np.int32)}
sess = tf.Session(graph=g1)
output = sess.run(e, feed_dict=replace)

print output

writer = tf.summary.FileWriter("./tf_graph", sess.graph)
writer.close()

sess.close()

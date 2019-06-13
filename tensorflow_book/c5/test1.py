import tensorflow as tf

#输入张量：[batch_size , height , width , channel]
input_batch = tf.constant([
    [
    [[0.0],[1.0],[2.0],[3.0],[4.0],[5.0]],
    [[0.1],[1.1],[2.1],[3.1],[4.1],[5.1]],
    [[0.2],[1.2],[2.2],[3.2],[4.2],[5.2]],
    [[0.3],[1.3],[2.3],[3.3],[4.3],[5.3]],
    [[0.4],[1.4],[2.4],[3.4],[4.4],[5.4]],
    [[0.5],[1.5],[2.5],[3.5],[4.5],[5.5]]
        ]
])

kernel = tf.constant([
    [[[0.0]],[[0.5]],[[1.0]]],
    [[[0.0]],[[1.0]],[[0.0]]],
    [[[0.0]],[[0.5]],[[0.0]]]
])

conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1,3,3,1], padding='SAME')

with tf.Session() as sess:
    #print input_batch
    #print sess.run(input_batch)
    #print kernel
    #print sess.run(kernel)
    print conv2d
    print sess.run(conv2d)


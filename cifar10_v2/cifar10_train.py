
import os
import numpy as np
import time
import datetime
from six.moves import xrange
import tensorflow as tf
import cifar10_data
import cifar10_model

global_step = tf.Variable(0, trainable=False)

# Get training images and labels for CIFAR-10.
# inputs img shape: [100, 24, 24, 3], data type: float32, range: [0, 1]
#        lbl shape: [100,], data type: int32, range: [0, 9]
images, labels = cifar10_data.distort_inputs('./input_data', cifar10_data.TRAIN_BATCH_SIZE)

# Build a Graph that computes the logits predictions from the inference model.
logits = cifar10_model.inference(images, 0.75, True)

# Calculate loss.
loss = cifar10_model.loss(logits, labels)

# Calculate accuracy.
eval_op = cifar10_model.eval(logits, labels)

# Build a Graph that trains the model with one batch of examples and
# updates the model parameters.
train_op = cifar10_model.train(loss, global_step)

# Create a saver.
saver = tf.train.Saver(tf.all_variables())

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()

# Init session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Get the summary writer.
summary_writer = tf.summary.FileWriter(cifar10_data.TRAIN_LOG_DIR, sess.graph)

# Make check point dir
if not os.path.exists(cifar10_data.CKPT_DIR):
    os.mkdir(cifar10_data.CKPT_DIR)

# Check if resore from last train
ckpt = tf.train.get_checkpoint_state(cifar10_data.CKPT_DIR)
if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0.xxx,
    # extract current step from it.
    current_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1].split('.')[0])
else:
    current_step = -1

# training loop
for step in xrange(current_step + 1, cifar10_data.MAX_TRAIN_STEPS):
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])
    duration = time.time() - start_time
    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step != 0 and step % 10 == 0:
        examples_per_sec = cifar10_data.TRAIN_BATCH_SIZE / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch))

    if step != 0 and step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

    # Save the model checkpoint periodically.
    if step != 0 and (step % 1000 == 0 or (step + 1) == cifar10_data.MAX_TRAIN_STEPS):
        checkpoint_path = os.path.join(cifar10_data.CKPT_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

summary_writer.close()
sess.close()


"""
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        op1 = [next_training_imgs, next_training_lbls]
        op2 = next_training_lbls
        (images1, labels1), labels2 = sess.run([op1, op2])

        print(labels1)
        print(labels2)
"""


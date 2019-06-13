
import tensorflow as tf
import cifar10_data
import cifar10_model
import numpy as np
from datetime import datetime
import time

# Get testing images and labels for CIFAR-10.
# inputs img shape: [batch_size, 24, 24, 3], data type: float32, range: [0, 1]
#        lbl shape: [batch_size,], data type: int32, range: [0, 9]
images, labels, iter = cifar10_data.inputs(
    True, './input_data', cifar10_data.EVAL_BATCH_SIZE)

# Build a Graph that computes the logits predictions from the inference model.
logits = cifar10_model.inference(images, 1.0, False)

# Calculate predictions.
top_k_op = tf.nn.in_top_k(logits, labels, 1)

# Init session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Create a saver.
# saver = tf.train.Saver()
# Restore the moving average version of the learned variables for eval.
variable_averages = tf.train.ExponentialMovingAverage(0.9999)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

# Get the summary writer.
summary_writer = tf.summary.FileWriter(cifar10_data.EVAL_LOG_DIR, sess.graph)

while True:
    ckpt = tf.train.get_checkpoint_state(cifar10_data.CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0.xxx,
        # extract global_step from it.
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1].split('.')[0])
    else:
        print('No checkpoint file found')
        time.sleep(200)
        continue

    # Re-initialize dataset iterator
    sess.run(iter.initializer)

    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = 0
    while True:
        try:
            predictions = sess.run(top_k_op)
            true_count += np.sum(predictions)
            total_sample_count += cifar10_data.EVAL_BATCH_SIZE
        except tf.errors.OutOfRangeError:
            break

    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ %d = %.3f' % (datetime.now(), global_step, precision))
    summary = tf.Summary()
    summary.value.add(tag='Precision @ 1', simple_value=precision)
    summary_writer.add_summary(summary, global_step)

    if global_step >= cifar10_data.MAX_TRAIN_STEPS - 1:
        break
    time.sleep(200)

summary_writer.close()
sess.close()

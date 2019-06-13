import tensorflow as tf
import cifar10_data
import cifar10_model
import datavis


# Get testing images and labels for CIFAR-10.
# inputs img shape: [batch_size, 24, 24, 3], data type: float32, range: [0, 1]
#        lbl shape: [batch_size,], data type: int32, range: [0, 9]
raw_images, images, labels, iter = cifar10_data.pred_inputs(
    True, './input_data', 10)

# Build a Graph that computes the logits predictions from the inference model.
logits = cifar10_model.inference(images, 1.0, False)

# Read label names
lbl_names = cifar10_data.read_lbl_names('./input_data')

# Calculate predictions.
pred_op = tf.argmax(logits, 1)

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

ckpt = tf.train.get_checkpoint_state(cifar10_data.CKPT_DIR)
if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0.xxx,
    # extract global_step from it.
    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1].split('.')[0])
else:
    print('No check points find.')
    exit(0)

# Re-initialize dataset iterator
sess.run(iter.initializer)


while True:
    try:
        raw_image_values, pred_values, label_values = sess.run([raw_images, pred_op, labels])
    except tf.errors.OutOfRangeError:
        break
    pred_lbls = ['p:' + lbl_names[int(i)] for i in pred_values]
    real_lbls = [',r:' + lbl_names[int(i)] for i in label_values]
    lbls = [val + real_lbls[i] for i,val in enumerate(pred_lbls)]
    datavis.data_vis(raw_image_values, lbls)
    str = input("Enter your input: ")
    if str != 'c':
        break
sess.close()

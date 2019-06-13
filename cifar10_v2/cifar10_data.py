
import tensorflow as tf
from six.moves import xrange
import datavis
import os

# Global const variable define
TRAIN_LOG_DIR = './train_logs'
EVAL_LOG_DIR = './eval_logs'
CKPT_DIR = './ckpts'
TRAIN_BATCH_SIZE = 100
EVAL_BATCH_SIZE = 100
CLASS_NUM = 10
MAX_TRAIN_STEPS = 150000

label_bytes = 1
image_height = 32
image_width = 32
image_depth = 3
image_bytes = image_height * image_width * image_depth
resized_height = 24
resized_width = 24


def _parse_single_example(raw_data):
    example_bytes = tf.decode_raw(raw_data, tf.uint8)
    label = tf.cast(tf.slice(example_bytes, [0], [label_bytes]), tf.int32)
    label = tf.squeeze(label, squeeze_dims=0)  # 降维为标量
    image = tf.slice(example_bytes, [label_bytes], [image_bytes])
    image = tf.transpose(tf.reshape(image, [image_depth, image_height, image_width]), [1, 2, 0])
    image = tf.cast(image, tf.float32)
    return image, label


def _parse(raw_data):
    image, label = _parse_single_example(raw_data)
    image = tf.image.resize_image_with_crop_or_pad(image, resized_height, resized_width)
    image = tf.image.per_image_standardization(image)
    return image, label


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
    for f in filenames:
        if not os.path.isfile(f):
            raise ValueError('Failed to find file: ' + f)

    fixed_len = label_bytes + image_bytes
    dataset = tf.data.FixedLengthRecordDataset(filenames, fixed_len)
    dataset = dataset.map(_parse)
    dataset = dataset.batch(batch_size)
    dataset_iter = dataset.make_initializable_iterator()
    next_imgs, next_lbls = dataset_iter.get_next()
    return next_imgs, next_lbls, dataset_iter


def _pred_parse(raw_data):
    raw_image, label = _parse_single_example(raw_data)
    image = tf.image.resize_image_with_crop_or_pad(raw_image, resized_height, resized_width)
    image = tf.image.per_image_standardization(image)
    return raw_image, image, label


def pred_inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
    for f in filenames:
        if not os.path.isfile(f):
            raise ValueError('Failed to find file: ' + f)

    fixed_len = label_bytes + image_bytes
    dataset = tf.data.FixedLengthRecordDataset(filenames, fixed_len)
    dataset = dataset.map(_pred_parse)
    dataset = dataset.batch(batch_size)
    dataset_iter = dataset.make_initializable_iterator()
    next_raw_imgs, next_imgs, next_lbls = dataset_iter.get_next()
    return next_raw_imgs, next_imgs, next_lbls, dataset_iter


def _distort_parse(raw_data):
    image, label = _parse_single_example(raw_data)

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(image, [24, 24, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta= 63.0)

    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    distorted_image = tf.image.per_image_standardization(distorted_image)

    return distorted_image, label


def distort_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not os.path.isfile(f):
            raise ValueError('Failed to find file: ' + f)

    fixed_len = label_bytes + image_bytes
    dataset = tf.data.FixedLengthRecordDataset(filenames, fixed_len)
    dataset = dataset.map(_distort_parse)
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()
    data_iter = dataset.make_one_shot_iterator()
    next_training_imgs, next_training_lbls = data_iter.get_next()
    return next_training_imgs, next_training_lbls


def read_lbl_names(data_dir):
    filename = os.path.join(data_dir, 'batches.meta.txt')
    if not os.path.isfile(filename):
        raise ValueError('Failed to find file: ' + filename)
    f = open(filename)
    lbls = [line.strip('\n') for line in f.readlines() if line != '\n']
    f.close()
    return lbls


if __name__ == '__main__':
    train_step = 2
    batch_size = 10
    print('--test cifar10_data--')
    print('test1:read_lbl_names')
    lbl_names = read_lbl_names('./input_data')
    print(lbl_names)

    next_train_images, next_train_labels = distort_inputs('./input_data', batch_size)

    eval_dataset = inputs(True, './input_data')
    eval_dataset = eval_dataset.batch(batch_size)
    eval_iter = eval_dataset.make_one_shot_iterator()
    next_eval_images, next_eval_labels = eval_iter.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in xrange(train_step):
            print('test2:train data')
            images, labels = sess.run([tf.cast(next_train_images, tf.uint8), next_train_labels])
            labels = [lbl_names[int(i)] for i in labels]
            datavis.data_vis(images, labels)

            #print('test3:eval data')
            #images, labels = sess.run([tf.cast(next_eval_images, tf.uint8), next_eval_labels])
            #labels = [lbl_names[int(i)] for i in labels]
            #datavis.data_vis(images, labels)

    # 注意：在对图片进行变换前需要将[0,255]的uint8像素取值变为[0.,255.]的float32型
    #      显示图片需要注释掉 per_image_standardization，因为标准化后像素均值为0，标准差为1，肯定有负值，无法作为图片显示
    #      显示图片前还需将[0.,255.]的float32型变为[0,255]的uint8像素

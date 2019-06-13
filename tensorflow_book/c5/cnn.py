# -*- coding:utf-8 -*-

import glob
from itertools import groupby
from collections import defaultdict
import tensorflow as tf
import os


def read_raw_image():
    image_filenames = glob.glob("./imagenet_dogs/images/n02*/*.jpg")

    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)

    # 将文件名分解为品种和相应的文件名，品种对应于文件夹名称
    image_filename_with_breed = map(lambda filename:
                                    (filename.split("/")[3], filename), image_filenames)

    # 依据品种（上述返回元组的第0个分量）对图像分组
    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        # 枚举每个品种的图像，并将大致20%的图像划入测试集

        for i, breed_image in enumerate(breed_images):
            if i % 5 == 0:
                testing_dataset[dog_breed].append(breed_image[1])
            else:
                training_dataset[dog_breed].append(breed_image[1])

        # 检查每个品种的测试图像是否至少有全部图像的18%
        breed_training_count = float(len(training_dataset[dog_breed]))
        breed_testing_count = float(len(testing_dataset[dog_breed]))
        assert round(breed_testing_count / (breed_testing_count + breed_training_count),
                     2) > 0.18, "Not enough testing images."

    return training_dataset, testing_dataset


def write_records_file(dataset, record_location):
    """
    用dataset中的图像填充一个TFRecord文件，并将其类别包含进来
    :param dataset: dict(list) 这个字典的键对应于其值中文件名列表对应的标签
    :param record_location: str 存储TFRecord输出的路径
    :return:
    """
    path = record_location[:record_location.rfind('/')]
    if not os.path.exists(path):
        os.makedirs(path)
    writer = None
    sess = tf.Session()

    #枚举dataset，因为当前索引用于对文件进行划分，每隔100幅图像，训练样本的信息就被写入到一个新的TFRecord文件中，以加快写操作的进程
    current_index = 0

    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1
            image_file = tf.read_file(image_filename)

            #在ImageNet的狗的图像中，有少量无法被TensorFlow识别为JPEG的图像，利用try/catch可将这些图像忽略
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print("type mismatch, ignore this image:{image_filename}".format(image_filename=image_filename))
                continue

            #转换为灰度图可减少处理的计算量和内存占用，但这并不是必须的
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, (250, 151))

            #这里之所以用tf.cast，是因为虽然尺寸更改后的图像的数据类型是浮点型，但RGB值尚未转换到[0, 1)区间内
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_label = breed.encode("utf-8")

            example = tf.train.Example(features=tf.train.Features(feature={
                'label':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())

    writer.close()
    sess.close()


def load_training_image_from_records_file(records_file):
    ###加载图像###
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(records_file))
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(serialized, features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
    })

    record_image = tf.decode_raw(features['image'], tf.uint8)

    # 修改图像的形状有助于训练和输出的可视化
    image = tf.reshape(record_image, [250, 151, 1])

    label = tf.cast(features['label'], tf.string)

    min_after_dequeue = 10
    batch_size = 3
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return label, batch_size, image_batch, label_batch



def load_testing_image_from_records_file(records_file):
    ###加载图像###
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(records_file))
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(serialized, features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string)
    })

    record_image = tf.decode_raw(features['image'], tf.uint8)

    # 修改图像的形状有助于训练和输出的可视化
    image = tf.reshape(record_image, [250, 151, 1])

    label = tf.cast(features['label'], tf.string)

    min_after_dequeue = 10
    batch_size = 3
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.batch(
        [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return label, batch_size, image_batch, label_batch




def make_image_model(label, batch_size, image_batch, label_batch):
    ###模型###

    #将图像转换为灰度值位于[0,1)的浮点类型，以与convolution2d期望的输入匹配
    float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

    conv2d_layer_one = tf.contrib.layers.conv2d(
        float_image_batch,
        num_output_channels=32,  #要生成的滤波器数量
        kernel_size=(5,5),       #滤波器的宽度和高度
        activation_fn=tf.nn.relu,
        weight_init=tf.random_normal,
        stride=(2,2),
        trainable=True)
    pool_layer_one = tf.nn.max_pool(conv2d_layer_one, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #注意，卷积输出的第1维和最后一维未发生改变，但中间的两位维发生了变化
    #conv2d_layer_one.get_shape(), pool_layer_one.get_shape()
    #这段代码执行后可得输出：
    #(TensorShape([Dimension(3), Dimension(125), Dimension(76), Dimension(32)]),
    #TensorShape([Dimension(3), Dimension(63), Dimension(38), Dimension(32)]))


    conv2d_layer_two = tf.contrib.layers.conv2d(
        pool_layer_one,
        num_output_channels=64,  # 更多的输出通道意味着滤波器数量的增加
        kernel_size=(5, 5),      # 滤波器的宽度和高度
        activation_fn=tf.nn.relu,
        weight_init=tf.random_normal,
        stride=(1, 1),
        trainable=True)
    pool_layer_two = tf.nn.max_pool(conv2d_layer_two, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    #conv2d_layer_two.get_shape(), pool_layer_two.get_shape()
    #这段代码执行后可得输出：
    #(TensorShape([Dimension(3), Dimension(63), Dimension(38), Dimension(64)]),
    #TensorShape([Dimension(3), Dimension(32), Dimension(19), Dimension(64)]))



    flattened_layer_two = tf.reshape(pool_layer_two,
                                     [
                                         batch_size, #image_batch中的每幅图像
                                         -1          #输入的其他所有维
                                     ])
    flattened_layer_two.get_shape()
    #执行后得：
    #TensorShape([Dimension(3), Dimension(38912)])

    #weight_init参数也可接收一个可调用参数，这里使用一个lambda表达式返回了一个截断的正态分布
    #并指定了该分布的标准差
    hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        512,
        weight_init=lambda i, dtype: tf.truncated_normal([38912, 512], stddev=0.1),
        activation_fn=tf.nn.relu)

    #对一些神经元进行dropout处理，削减他们在模型中的重要性
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

    #输出是前面的层与训练中可用的120个不同的狗的品种的全连接
    final_fully_connected = tf.contrib.layers.fully_connected(
        hidden_layer_three,
        120, #ImageNet Dogs数据集中狗的品种数
        weight_init=lambda i, dtype: tf.truncated_normal([512, 120], stddev=0.1))


    #找到位于imagenet-dogs路径下的所有目录名（n02085620-Chihuahua,...）
    labels = list(map(lambda c: c.split("/")[-1], glob.glob("./imagenet-dogs/images/*")))

    #匹配每个来自label_batch的标签并返回它们在类别列表中的索引
    train_labels = tf.map_fn(lambda l: tf.where(tf.equal(label, l)), label_batch, dtype=tf.int64)

    return hidden_layer_three, final_fully_connected, train_labels  # X, tf.matmul(X, W) + b, Y






########################################################

#初始化变量和模型参数，定义训练闭环中的运算
feature_number = 512 #
class_number = 120 #狗类别数


def inference(combine_inputs):
    """ 计算推断模型在数据X上的输出，并将结果返回

    """
    return tf.nn.softmax(combine_inputs)

def loss(combine_inputs, Y):
    """ 依据训练数据X及其期望输出Y计算损失

    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs, labels=Y))


def train(total_loss):
    """ 依据计算的总损失训练或调整模型参数
    :param total_loss:
    :return:
    """
    learning_rate = 0.1
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess):
    """ 对训练得到的模型进行评估
    :param sess:
    :param X:
    :param Y:
    :return:
    """
    # todo load_testing_image_from_records_file
    label, batch_size, image_batch, label_batch = load_testing_image_from_records_file("./output/testing-images/*.tfrecords")
    X, combine_inputs, Y = make_image_model(label, batch_size, image_batch, label_batch)

    predicted = tf.cast(tf.argmax(inference(combine_inputs), 1), tf.int32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))



if  __name__ == "__main__":
    training_dataset, testing_dataset = read_raw_image()
    write_records_file(training_dataset, "./output/training-images/training-image")
    print('write training records file done!')
    write_records_file(testing_dataset, "./output/testing-images/testing-image")
    print('write testing records file done!')
    label, batch_size, image_batch, label_batch = load_training_image_from_records_file("./output/training-images/*.tfrecords")
    print('load_training_image_from_records_file done!')
    X, combine_inputs, Y = make_image_model(label, batch_size, image_batch, label_batch)
    print('make_image_model done!')

    #创建一个Saver对象
    #saver = tf.train.Saver()

    #在一个会话对象中启动数据流图，搭建流程
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        #X, Y = hidden_layer_three, train_labels  # 3*512  3*1

        total_loss = loss(combine_inputs, Y)
        train_op = train(total_loss)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        '''
        for i in range(feature_number):
            for j in range(class_number):
                name = 'W%d%d' % (i, j)
                tf.summary.scalar(bytes(name), W[i][j])
        for i in range(class_number):
            name = 'b%d' % i
            tf.summary.scalar(bytes(name), b[i])

        tf.summary.scalar(b'loss', total_loss)
        '''
        merged_summaries = tf.summary.merge_all()

        writer = tf.summary.FileWriter('./c5/cnn-graph', graph=sess.graph)

        # 实际的训练迭代次数
        train_steps = 3000
        for step in range(train_steps):
            _, summary = sess.run([train_op, merged_summaries])
            writer.add_summary(summary, global_step=step)


        #print sess.run(W)
        #print sess.run(b)
        evaluate(sess)

        coord.request_stop()
        coord.join(threads)


        writer.flush()
        writer.close()
        sess.close()
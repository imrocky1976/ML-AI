"""Functions for reading MNIST data."""

import numpy as np
import collections
import io
import os

Datasets = collections.namedtuple('Datasets', ['training_image', 'training_label',
                                              'testing_image', 'testing_label'])


def read_image_data(image_data_path, reshape=False):
    """
    Read the image into a 4D uint8 numpy array [index, y, x, depth].
    :params:
     image_data_path: The image file full path for reading.
     reshape: if True, reshape images shape to [num_images, rows * cols].
    :return A 4D float32 numpy array [index, y, x, depth] if reshape is False, else return
            a 2D float32 numpy array [index, y * x].
    :raise ValueError: If the file does not start with 2051.
    """
    image_file = io.open(image_data_path, 'rb')
    try:
        uint_dt = np.dtype(np.uint32).newbyteorder('>')
        magic = np.frombuffer(image_file.read(4), uint_dt)[0]
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, image_data_path))
        num_images = np.frombuffer(image_file.read(4), uint_dt)[0]
        rows = np.frombuffer(image_file.read(4), uint_dt)[0]
        cols = np.frombuffer(image_file.read(4), uint_dt)[0]

        buf = image_file.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
    finally:
        image_file.close()
    data = data.reshape(num_images, rows, cols, 1)
    #[0,255] -> [0.0,1.0]
    data = np.float32(data) / 255
    print('images data shape:')
    print(np.shape(data))
    #print('first image data:')
    #print(data[0])
    #print('last image data:')
    #print(data[num_images - 1])
    if reshape:
        data = data.reshape(num_images, rows * cols)
        print('images data reshaped:')
        print(np.shape(data))
    return data


def read_label_data(label_data_path):
    """
    Read label data into a 1D uint8 numpy array [index].
    :param label_data_path: The label data file full path for reading.
    :return A 1D uint8 numpy array [index]
    :raise ValueError: If the file does not start with 2049.
    """
    with io.open(label_data_path, 'rb') as label_file:
        uint_dt = np.dtype(np.uint32).newbyteorder('>')
        magic = np.frombuffer(label_file.read(4), uint_dt)[0]
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, label_data_path))
        num_labels = np.frombuffer(label_file.read(4), uint_dt)[0]

        buf = label_file.read(num_labels)
        data = np.frombuffer(buf, dtype=np.uint8)
    print('label data shape:')
    print(np.shape(data))
    #print('first label data:')
    #print(data[0])
    #print('last label data:')
    #print(data[num_labels - 1])
    return data


def read_data_sets(data_dir, reshape=False):
    """
    Read training data and testing data.
    :params:
     data_dir: Data directory.
     reshape: if True, reshape images data shape to [num_images, rows * cols].
    :return A named tuple called "Datasets": ('training_image', 'training_label'
                                        'testing_image', 'testing_label')
    """
    print('Read training data:')
    training_image = read_image_data(os.path.join(data_dir, 'train-images-idx3-ubyte/data'), reshape=reshape)
    training_label = read_label_data(os.path.join(data_dir, 'train-labels-idx1-ubyte/data'))

    print('Read testing data:')
    testing_image = read_image_data(os.path.join(data_dir, 't10k-images-idx3-ubyte/data'), reshape=reshape)
    testing_label = read_label_data(os.path.join(data_dir, 't10k-labels-idx1-ubyte/data'))

    return Datasets(training_image=training_image, training_label=training_label,
                    testing_image=testing_image, testing_label=testing_label)


def random_batch(images, labels, batch_size):
    assert np.shape(images)[0] == np.shape(labels)[0], \
        'Labels shape(%d) does not match images shape(%d)' % (np.shape(images)[0], np.shape(labels)[0])

    num_examples = np.shape(labels)[0]
    assert batch_size <= num_examples, 'Batch size(%d) is bigger than example number(%d)' % (batch_size, num_examples)

    indexes = np.arange(num_examples)
    np.random.shuffle(indexes)
    batch_images = images[indexes[:batch_size]]
    batch_labels = labels[indexes[:batch_size]]
    #print('Random batch indexes:')
    #print(indexes[:batch_size])
    return batch_images, batch_labels
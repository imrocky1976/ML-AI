"""Functions for reading and pre-processing the CIFAR10 data before train."""

import os
import tarfile
import numpy as np
import pickle
import datavis
import tensorflow as tf


class Cifar10Data(object):

    NUM_CLASS = 10
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGE_DEPTH = 3
    __NUM_CASES_PER_BATCH = 10000

    def __init__(self, data_dir='./input_data', dtype=np.float32):
        self.__testing_batch_count = 0
        self.__data_dir = data_dir
        self.__data_type = dtype
        self.__label_names, \
            self.__testing_images, \
            self.__testing_labels, \
            self.__training_images, \
            self.__training_labels = self.__inputs()

    def __inputs(self):
        if not os.path.isfile('cifar-10-batches-py/batches.meta') \
                and os.path.isfile('cifar-10-batches-py/test_batch') \
                and os.path.isfile('cifar-10-batches-py/data_batch_*'):
            self.__extract_data('cifar-10-python.tar.gz')

        label_data = self.__unpickle('cifar-10-batches-py/batches.meta')
        label_names = label_data[b'label_names']

        testing_data = self.__unpickle('cifar-10-batches-py/test_batch')
        # [10000, 3 * 32 * 32] --> [10000, 3, 32, 32]
        testing_images = np.reshape(testing_data[b'data'],
                                    [self.__NUM_CASES_PER_BATCH, self.IMAGE_DEPTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH])
        # [10000, 3, 32, 32] --> [10000, 32, 32, 3]
        testing_images = np.transpose(testing_images, [0, 2, 3, 1])
        testing_labels = np.array(testing_data[b'labels'])

        training_images = None
        training_labels = None
        for i in range(1, 6):
            training_data = self.__unpickle('cifar-10-batches-py/data_batch_%d' % i)
            # [10000, 3 * 32 * 32] --> [10000, 3, 32, 32]
            temp_images = np.reshape(training_data[b'data'], [self.__NUM_CASES_PER_BATCH, self.IMAGE_DEPTH,
                                                             self.IMAGE_HEIGHT, self.IMAGE_WIDTH])
            # [10000, 3, 32, 32] --> [10000, 32, 32, 3]
            if training_images is None:
                training_images = np.transpose(temp_images, [0, 2, 3, 1])
            else:
                training_images = np.concatenate((training_images, np.transpose(temp_images, [0, 2, 3, 1])))
            if training_labels is None:
                training_labels = training_data[b'labels']
            else:
                training_labels = np.concatenate((training_labels, training_data[b'labels']))

        if self.__data_type == np.float32:
            testing_images = np.float32(testing_images) / 255.0
            training_images = np.float32(training_images) / 255.0

        assert np.shape(testing_images)[0] == np.shape(testing_labels)[0], \
            'Testing labels shape(%d) does not match testing images shape(%d)' % \
            (np.shape(testing_images)[0], np.shape(testing_labels)[0])
        assert np.shape(training_images)[0] == np.shape(training_labels)[0], \
            'Training labels shape(%d) does not match training images shape(%d)' % \
            (np.shape(training_labels)[0], np.shape(training_images)[0])

        return label_names, testing_images, testing_labels, training_images, training_labels

    def __extract_data(self, fname):
        fpath = os.path.join(self.__data_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError('File not found: %s' % fpath)
        print('Extracting', fpath)
        tar = tarfile.open(fpath, 'r:gz')
        tar.extractall(self.__data_dir)
        tar.close()

    def __unpickle(self, fname):
        fpath = os.path.join(self.__data_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError('File not found: %s' % fpath)
        print('Unpickling', fpath)
        with open(fpath, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    @property
    def training_images(self):
        return self.__training_images

    @property
    def training_labels(self):
        return self.__training_labels

    @property
    def testing_images(self):
        return self.__testing_images

    @property
    def testing_labels(self):
        return self.__testing_labels

    def label2name(self, label_indices):
        return [self.__label_names[i] for i in label_indices]

    @property
    def label_names(self):
        return self.__label_names

    def init_testing_batch(self):
        self.__testing_batch_count = 0

    def next_testing_batch(self, max_batch_size):
        num_examples = self.__testing_images.shape[0]
        assert max_batch_size <= num_examples, \
            'Max batch size(%d) is bigger than testing example number(%d)' % (max_batch_size, num_examples)

        low_index = max_batch_size * self.__testing_batch_count
        assert 0 <= low_index < self.__testing_images.shape[0], 'Index error!'
        high_index = max_batch_size * (self.__testing_batch_count + 1)
        if high_index > self.__testing_images.shape[0]:
            high_index = self.__testing_images.shape[0]

        if high_index >= self.__testing_images.shape[0]:
            has_next = False
            self.__testing_batch_count = 0
        else:
            has_next = True
            self.__testing_batch_count += 1

        return has_next, self.__testing_images[low_index:high_index], self.__testing_labels[low_index:high_index]

    def random_training_batch(self, batch_size):
        num_examples = np.shape(self.__training_images)[0]
        assert batch_size <= num_examples, 'Batch size(%d) is bigger than training example number(%d)' % (
            batch_size, num_examples)

        indexes = np.arange(num_examples)
        np.random.shuffle(indexes)
        batch_images = self.__training_images[indexes[:batch_size]]
        batch_labels = self.__training_labels[indexes[:batch_size]]
        return batch_images, batch_labels


if __name__ == '__main__':
    cifar10_data = Cifar10Data('./input_data', np.uint8)
    print('testing_labels shape:')
    print(cifar10_data.testing_labels.shape)
    print('training_labels shape:')
    print(cifar10_data.training_labels.shape)
    has_next, testing_images, testing_labels = cifar10_data.next_testing_batch(10)
    print('has_next=%s' % str(has_next))
    print('testing_labels:')
    print(testing_labels)
    training_images, training_labels = cifar10_data.random_training_batch(10)
    print('training_labels:')
    print(training_labels)
    print('label_names:')
    print(cifar10_data.label2name([i for i in range(10)]))

    datavis.data_vis(training_images, cifar10_data.label2name([i for i in training_labels]))


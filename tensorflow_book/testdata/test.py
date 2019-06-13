# -*- coding:utf-8 -*-

import tensorflow as tf
import os

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__)
                                                     + '/' + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


passenger_id, survived, pclass, name, sex, age, sibsp, \
parch, ticket, fare, cabin, embarked = read_csv(1, 'test.csv', [[0.], [0.],
                                                                       [0], [""], [""],
                                                                       [0.], [0.], [0.],
                                                                       [""], [0.], [""],
                                                                       [""]])
is_first_class = tf.to_float(tf.equal(pclass, [1]))
is_second_class = tf.to_float(tf.equal(pclass, [2]))
is_third_class = tf.to_float(tf.equal(pclass, [3]))

gender = tf.to_float(tf.equal(sex, ["female"]))

#features = tf.transpose([is_first_class, is_second_class, is_third_class, gender, age])
#survived = tf.reshape(survived, [1, 1]) #418
features = [is_first_class, is_second_class, is_third_class, gender, age]




#if __name__ == "__main__":

sv = tf.train.Supervisor()
with sv.managed_session() as sess:
    X = features
    Y = survived
    print "Y"
    print sess.run(Y)
    print "YY"


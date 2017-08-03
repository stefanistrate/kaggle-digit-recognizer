#!/usr/bin/env python

import gflags as flags
import glog as log
import numpy as np
import pandas as pd
import sys
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_boolean(
        'train', False,
        'Whether to train a new model instead of running a saved one.')
flags.DEFINE_integer('num_training_steps', 50001,
                     'Number of steps to train the model.')
flags.DEFINE_integer('num_examples_per_batch', 50,
                     'Number of examples to use in a training batch.')
flags.DEFINE_string('tf_writer_suffix', 'cnn',
                    'Suffix for the TF FileWriter.')


def load_data():
    data = {}

    if FLAGS.train:
        log.info('Loading train data.')
        train = pd.read_csv('data/train.csv')

        log.info('Shuffling train data.')
        shuffled_train = train.sample(frac=1, random_state=7).as_matrix()

        log.info('Splitting train data into 95% for training and 5% for '
                 'evaluation.')
        num_training_examples = int(0.95 * shuffled_train.shape[0])
        train_x = shuffled_train[:num_training_examples, 1:].astype('float32')
        train_labels = np.eye(10)[shuffled_train[:num_training_examples, 0]]
        data['train'] = {'x': train_x, 'labels': train_labels}
        eval_x = shuffled_train[num_training_examples:, 1:].astype('float32')
        eval_labels = np.eye(10)[shuffled_train[num_training_examples:, 0]]
        data['eval'] = {'x': eval_x, 'labels': eval_labels}
    else:
        log.info('Loading test data.')
        test_x = pd.read_csv('data/test.csv').as_matrix().astype('float32')
        data['test'] = {'x': test_x}

    return data


def get_data_batch(data, batch_number, examples_per_batch):
    batch_start = (batch_number * examples_per_batch) % data['x'].shape[0]
    batch_end = batch_start + examples_per_batch
    return (data['x'][batch_start : batch_end],
            data['labels'][batch_start : batch_end])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='w')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_layer(x, w_shape, b_shape, name):
    with tf.name_scope(name):
        w = weight_variable(w_shape)
        b = bias_variable(b_shape)
        act = tf.nn.relu(conv2d(x, w) + b)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return max_pool_2x2(act)


def fc_layer(x, w_shape, b_shape, name):
    with tf.name_scope(name):
        w = weight_variable(w_shape)
        b = bias_variable(b_shape)
        h = tf.matmul(x, w) + b
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        return h


def construct_network(x, y_, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv_layer(x_image, [5, 5, 1, 32], [32], 'conv1')
    conv2 = conv_layer(conv1, [5, 5, 32, 64], [64], 'conv2')
    conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc1 = fc_layer(conv2_flat, [7 * 7 * 64, 1024], [1024], 'fc1')
    fc1_dropout = tf.nn.dropout(tf.nn.relu(fc1), keep_prob)
    fc2 = fc_layer(fc1_dropout, [1024, 10], [10], 'fc2')

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc2))
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return fc2, train_step, accuracy


if __name__ == '__main__':
    FLAGS(sys.argv)
    data = load_data()

    log.info('Constructing the convolutional neural network.')
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    network, train_step, accuracy = construct_network(x, y_, keep_prob)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if FLAGS.train:
            writer = tf.summary.FileWriter('/tmp/kaggle-digit-recognizer/%s'
                                           % FLAGS.tf_writer_suffix)
            writer.add_graph(sess.graph)

            log.info('Training CNN model...')
            sess.run(tf.global_variables_initializer())

            for i in range(FLAGS.num_training_steps):
                if i % 100 == 0:
                    log.info('step %d' % i)
                    s = sess.run(merged_summary,
                                 feed_dict={x: data['eval']['x'],
                                            y_: data['eval']['labels'],
                                            keep_prob: 1.0})
                    writer.add_summary(s, i)

                train_x, train_labels = get_data_batch(data['train'], i, 50)
                sess.run(train_step, feed_dict={x: train_x,
                                                y_: train_labels,
                                                keep_prob: 0.5})

            log.info('Evaluation accuracy: %g'
                     % accuracy.eval(feed_dict={x: data['eval']['x'],
                                                y_: data['eval']['labels'],
                                                keep_prob: 1.0}))

            save_path = saver.save(sess, 'models/cnn.ckpt')
            log.info('CNN model saved to %s.' % save_path)
        else:
            log.info('Restoring CNN model.')
            saver.restore(sess, 'models/cnn.ckpt')
            log.info('CNN model restored.')

            log.info('Predicting output labels...')
            prediction = tf.argmax(network, 1)
            classified = sess.run(prediction,
                                  feed_dict={x: data['test']['x'],
                                             keep_prob: 1.0})
            log.info('Done.')

            log.info('Saving output predictions.')
            output = pd.DataFrame({'Label' : pd.Series(classified)})
            output.index += 1
            output.index.name = 'ImageId'
            output.to_csv('outputs/cnn.txt')
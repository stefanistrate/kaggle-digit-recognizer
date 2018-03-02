#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import tensorflow as tf

from model import cnn_model_fn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
        'training_steps', 20000,
        'Number of training steps.')
tf.app.flags.DEFINE_integer(
        'training_batch_size', 100,
        'Number of training steps.')

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Load input data and split it into train & eval.
    df = pd.read_csv('data/train.csv', dtype=np.uint8)
    train_df = df.sample(frac=0.9, random_state=777)
    train_data = train_df.iloc[:, 1:].values.astype(np.float32)
    train_labels = train_df.iloc[:, 0].values.astype(np.int32)
    eval_df = df.drop(train_df.index)
    eval_data = eval_df.iloc[:, 1:].values.astype(np.float32)
    eval_labels = eval_df.iloc[:, 0].values.astype(np.int32)

    # Construct file paths.
    unique_name = (
            ('CNN_GradientDescent'
             '_TrainingSteps_%d'
             '_BatchSize_%d'
             '_Conv_%dx%d_%d'
             '_Pool_%dx%d_%d'
             '_Conv_%dx%d_%d'
             '_Pool_%dx%d_%d'
             '_Dense_%d'
             '_DropoutRate_%f'
             '_LearningRate_%f')
            % (FLAGS.training_steps,
               FLAGS.training_batch_size,
               FLAGS.conv1_kernel_size, FLAGS.conv1_kernel_size,
               FLAGS.conv1_filters,
               FLAGS.pool1_pool_size, FLAGS.pool1_pool_size,
               FLAGS.pool1_strides,
               FLAGS.conv2_kernel_size, FLAGS.conv2_kernel_size,
               FLAGS.conv2_filters,
               FLAGS.pool2_pool_size, FLAGS.pool2_pool_size,
               FLAGS.pool2_strides,
               FLAGS.dense_units,
               FLAGS.dropout_rate,
               FLAGS.learning_rate)
    )
    model_dir = 'models/%s' % unique_name
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    output_txt = '%s/%s.txt' % (outputs_dir, unique_name)

    # Create the estimator.
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir=model_dir)

    # Train the model.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=FLAGS.training_batch_size,
            num_epochs=None,
            shuffle=True)
    classifier.train(input_fn=train_input_fn,
                     steps=FLAGS.training_steps)

    # Evaluate the model.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    # Predict with the model.
    predict_data = (pd.read_csv('data/test.csv', dtype=np.uint8)
                    .values.astype(np.float32))
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': predict_data},
            num_epochs=1,
            shuffle=False)
    predict_results = classifier.predict(input_fn=predict_input_fn)

    # Write predictions.
    predict_classes = [result['class'] for result in predict_results]
    output_df = pd.DataFrame({'Label' : pd.Series(predict_classes)})
    output_df.index += 1
    output_df.index.name = 'ImageId'
    output_df.to_csv(output_txt)


if __name__ == '__main__':
    tf.app.run()

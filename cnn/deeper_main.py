#!/usr/bin/env python

import imgaug.augmenters as iaa
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from deeper_model import model_fn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
        'training_steps', 5000,
        'Number of training steps.')
tf.app.flags.DEFINE_integer(
        'batch_size', 100,
        'Training batch size.')

tf.logging.set_verbosity(tf.logging.INFO)


def augment_data(data):
    data = data.reshape(data.shape[0], 28, 28)

    seq = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.Multiply((0.9, 1.1)),
            iaa.Affine(
                    scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-10, 10),
            ),
    ], random_order=True)

    return seq.augment_images(data).reshape(data.shape[0], 784)


def normalize_data(data, mean, std):
    with np.errstate(divide='ignore', invalid='ignore'):
        data = (data - mean) / std
        data[~np.isfinite(data)] = 0.0
        return data


def main(unused_argv):
    # Load input data, split it into train & eval and augment the train part.
    df = pd.read_csv('data/train.csv', dtype=np.uint8)
    train_df = df.sample(frac=0.9, random_state=777)
    train_data = train_df.iloc[:, 1:].values
    train_data = np.concatenate([augment_data(train_data) for _ in range(5)])
    train_data = train_data.astype(np.float32)
    data_mean = np.mean(train_data, axis=0)
    data_std = np.std(train_data, axis=0)
    train_data = normalize_data(train_data, data_mean, data_std)
    train_labels = train_df.iloc[:, 0].values.astype(np.int32)
    train_labels = np.concatenate([train_labels for _ in range(5)])
    eval_df = df.drop(train_df.index)
    eval_data = eval_df.iloc[:, 1:].values.astype(np.float32)
    eval_data = normalize_data(eval_data, data_mean, data_std)
    eval_labels = eval_df.iloc[:, 0].values.astype(np.int32)

    # Construct file paths.
    unique_name = (
            ('CNN_DEEPER'
             '_GradientDescent'
             '_BatchSize_%d'
             '_LearningRate_%f')
            % (FLAGS.batch_size,
               FLAGS.learning_rate)
    )
    model_dir = 'models/%s' % unique_name
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    output_txt = '%s/%s.{}.txt' % (outputs_dir, unique_name)

    # Create the estimator.
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_dir)

    # Train the model.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=FLAGS.batch_size,
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
    predict_data = normalize_data(predict_data, data_mean, data_std)
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
    output_df.to_csv(output_txt.format(eval_results['global_step']))


if __name__ == '__main__':
    tf.app.run()

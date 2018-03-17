import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
        'conv1_filters', 32,
        'Number of output filters for conv1 layer.')
tf.app.flags.DEFINE_integer(
        'conv1_kernel_size', 5,
        'Kernel size for conv1 layer.')
tf.app.flags.DEFINE_integer(
        'pool1_pool_size', 2,
        'Max pooling filter size for pool1 layer.')
tf.app.flags.DEFINE_integer(
        'pool1_strides', 2,
        'Stride size for pool1 layer.')
tf.app.flags.DEFINE_integer(
        'conv2_filters', 64,
        'Number of output filters for conv2 layer.')
tf.app.flags.DEFINE_integer(
        'conv2_kernel_size', 5,
        'Kernel size for conv2 layer.')
tf.app.flags.DEFINE_integer(
        'pool2_pool_size', 2,
        'Max pooling filter size for pool2 layer.')
tf.app.flags.DEFINE_integer(
        'pool2_strides', 2,
        'Stride size for pool2 layer.')
tf.app.flags.DEFINE_integer(
        'dense_units', 1024,
        'Number of output units for the dense layer.')
tf.app.flags.DEFINE_float(
        'dropout_rate', 0.4,
        'Dropout rate to apply to the dense layer.')
tf.app.flags.DEFINE_string(
        'optimizer', 'GradientDescent',
        'TF optimizer algorithm to use.')
tf.app.flags.DEFINE_float(
        'learning_rate', 0.1,
        'Learning rate for the optimizer.')


def model_fn(features, labels, mode):
    # Input layer. Reshape pixel features to
    # [batch_size, image_width, image_height, num_color_channels].
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Convolutional layer #1.
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=FLAGS.conv1_filters,
                             kernel_size=FLAGS.conv1_kernel_size,
                             padding='same',
                             activation=tf.nn.relu)

    # Pooling layer #1.
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=FLAGS.pool1_pool_size,
                                    strides=FLAGS.pool1_strides)

    # Convolutional layer #2.
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=FLAGS.conv2_filters,
                             kernel_size=FLAGS.conv2_kernel_size,
                             padding='same',
                             activation=tf.nn.relu)

    # Pooling layer #2.
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=FLAGS.pool2_pool_size,
                                    strides=FLAGS.pool2_strides)

    # Flatten pooling layer #2.
    num_reduced_dim = int(28 / FLAGS.pool1_strides / FLAGS.pool2_strides)
    pool2_flat = tf.reshape(
            pool2,
            [-1, num_reduced_dim * num_reduced_dim * FLAGS.conv2_filters])

    # Dense layer.
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=FLAGS.dense_units,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=FLAGS.dropout_rate,
                                training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer.
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Generate predictions (for PREDICT and EVAL mode). Also add
    # `softmax_tensor` to the graph.
    predictions = {
        'class': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes).
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(
                    learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=FLAGS.learning_rate,
                    epsilon=1)
        elif FLAGS.optimizer == 'Ftrl':
            optimizer = tf.train.FtrlOptimizer(
                    learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'ProximalAdagrad':
            optimizer = tf.train.ProximalAdagradOptimizer(
                    learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'ProximalGradientDescent':
            optimizer = tf.train.ProximalGradientDescentOptimizer(
                    learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=FLAGS.learning_rate,
                    epsilon=1)
        else:
            optimizer = None
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode).
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels,
                                        predictions=predictions['class'])
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

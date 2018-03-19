import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float(
        'learning_rate', 0.1,
        'Learning rate for the optimizer.')


def _build_conv_group(input_layer, filters, kernel_size, mode):
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=filters,
                             kernel_size=kernel_size,
                             padding='same',
                             activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1,
                             filters=filters,
                             kernel_size=kernel_size,
                             padding='same',
                             activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(inputs=conv2,
                                   pool_size=2,
                                   strides=2)
    dropout = tf.layers.dropout(inputs=pool,
                                rate=0.25,
                                training=(mode == tf.estimator.ModeKeys.TRAIN))
    return dropout


def model_fn(features, labels, mode):
    # Input layer. Reshape pixel features to
    # [batch_size, image_width, image_height, num_color_channels].
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Convolutional layers.
    conv_group1 = _build_conv_group(input_layer, 32, 5, mode)
    conv_group2 = _build_conv_group(conv_group1, 64, 3, mode)

    # Dense layer.
    flatten = tf.reshape(conv_group2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=flatten,
                            units=256,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.5,
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
        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=FLAGS.learning_rate)
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

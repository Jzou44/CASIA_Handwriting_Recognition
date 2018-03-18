import tensorflow as tf
import config


def cnn(X, Y):
    input_layer = tf.reshape(X, [-1, config.image_hight, config.image_width, 1])

    # Input Tensor Shape: [batch_size, 128, 128, 1]
    # Output Tensor Shape: [batch_size, 128, 128, 32]
    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.leaky_relu)

    # Input Tensor Shape: [batch_size, 128, 128, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    with tf.name_scope("pool1"):
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 64]
    with tf.name_scope("conv2"):
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.leaky_relu)

    # Input Tensor Shape: [batch_size, 64, 64, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    with tf.name_scope("pool2"):
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 128]
    with tf.name_scope("conv3"):
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.leaky_relu)

    # Input Tensor Shape: [batch_size, 32, 32, 128]
    # Output Tensor Shape: [batch_size, 16, 16, 128]
    with tf.name_scope("pool3"):
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 16, 16, 128]
    # Output Tensor Shape: [batch_size, 16, 16, 256]
    with tf.name_scope("conv4"):
        conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.leaky_relu)

    # Input Tensor Shape: [batch_size, 16, 16, 256]
    # Output Tensor Shape: [batch_size, 8, 8, 256]
    with tf.name_scope("pool4"):
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 8, 8, 256]
    # Output Tensor Shape: [batch_size, 8, 8, 256]
    with tf.name_scope("conv5"):
        conv5 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.leaky_relu)

    # Input Tensor Shape: [batch_size, 8, 8, 256]
    # Output Tensor Shape: [batch_size, 4, 4, 256]
    with tf.name_scope("pool5"):
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 4, 4, 256]
    # Output Tensor Shape: [batch_size, 4, 4, 512]
    with tf.name_scope("conv6"):
        conv6 = tf.layers.conv2d(inputs=pool5, filters=512, kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.leaky_relu)

    # Input Tensor Shape: [batch_size, 4, 4, 512]
    # Output Tensor Shape: [batch_size, 2, 2, 512]
    with tf.name_scope("pool6"):
        pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    # Input Tensor Shape: [batch_size, 2, 2, 512]
    # Output Tensor Shape: [batch_size, 2 * 2 * 512]
    feature_layer = tf.reshape(pool6, [-1, 2 * 2 * 512])

    return feature_layer

def full_connected_classifier(feature_layer):
    with tf.name_scope("dense1"):
        dense1 = tf.layers.dense(inputs=feature_layer, units=1024, activation=tf.nn.leaky_relu)
    with tf.name_scope("dropout1"):
        dropout1 = tf.layers.dropout(dense1, rate=0.25, training=config.MODE == tf.estimator.ModeKeys.TRAIN)
    with tf.name_scope("dense2"):
        dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.leaky_relu)
    with tf.name_scope("dropout2"):
        dropout2 = tf.layers.dropout(dense2, rate=0.25, training=config.MODE == tf.estimator.ModeKeys.TRAIN)
    with tf.name_scope("out_layer"):
        classification_layer = tf.layers.dense(inputs=dropout2, units=config.label_array_length, activation=tf.nn.leaky_relu)

    return classification_layer

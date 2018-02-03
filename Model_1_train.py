import tensorflow as tf
import numpy as np
import config as config
import CASIA_data_manager as dataManager
from Logger import log

# Parameters
# learning_rate = 0.005
# training_epochs = 2
# batch_size = 100
# display_step = 55
# n_input = 784  # MNIST data input (img shape: 28*28)
# n_classes = 100  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, config.target_image_hight, config.target_image_width])
Y = tf.placeholder("float", [None, config.subset_label_array_length])

input_layer = tf.reshape(X, [-1, config.target_image_hight, config.target_image_width, 1])

# Input Tensor Shape: [batch_size, 128, 128, 1]
# Output Tensor Shape: [batch_size, 64, 64, 1]
pool0 = tf.layers.max_pooling2d(inputs=input_layer, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 64, 64, 1]
# Output Tensor Shape: [batch_size, 64,64, 32]
conv1 = tf.layers.conv2d(
    inputs=pool0,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 64,64, 32]
# Output Tensor Shape: [batch_size, 32, 32, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 32, 32, 32]
# Output Tensor Shape: [batch_size, 32, 32, 64]
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 32, 32, 64]
# Output Tensor Shape: [batch_size, 16, 16, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 16, 16, 64]
# Output Tensor Shape: [batch_size, 16, 16, 128]
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 16, 16, 128]
# Output Tensor Shape: [batch_size, 8, 8, 128]
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 8, 8, 128]
# Output Tensor Shape: [batch_size, 8, 8, 256]
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 8, 8, 256]
# Output Tensor Shape: [batch_size, 8, 8, 256]
conv5 = tf.layers.conv2d(
    inputs=conv4,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 8, 8, 256]
# Output Tensor Shape: [batch_size, 4, 4, 256]
pool4 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 4, 4, 256]
# Output Tensor Shape: [batch_size, 4, 4, 256]
conv6 = tf.layers.conv2d(
    inputs=pool4,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 4, 4, 256]
# Output Tensor Shape: [batch_size, 2, 2, 256]
pool5 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 2, 2, 256]
# Output Tensor Shape: [batch_size, 2 * 2 * 256]
pool5_flat = tf.reshape(pool5, [-1, 2 * 2 * 256])

dense1 = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu)
out_layer = tf.layers.dense(inputs=dense1, units=config.subset_label_array_length, activation=tf.nn.relu)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "weigths/model_1/weigths.ckpt")
    dm = dataManager.DataManager()

    batch_round = 0
    while True:
        batch_x, batch_y = dm.next_batch(config.batch_size)
        _, cost = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        log.info("batch_round: %d" % (batch_round))
        log.info("cost={:.9f}".format(cost))
        if np.mod(batch_round, 100) == 0:
            save_path = saver.save(sess, "weigths/model_1/weigths.ckpt")
            log.info('save model 1 weight to weigths/model_1/weigths.ckpt')
        batch_round = batch_round + 1
    print("Optimization Finished!")

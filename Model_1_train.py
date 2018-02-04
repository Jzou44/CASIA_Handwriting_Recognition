import tensorflow as tf
import numpy as np
import config as config
import CASIA_data_manager as dataManager
from Logger import log

weigth_path = 'weigths/model_1/weigths.ckpt'

# tf Graph input
X = tf.placeholder("float", [None, config.target_image_hight, config.target_image_width])
Y = tf.placeholder("float", [None, config.subset_label_array_length])

input_layer = tf.reshape(X, [-1, config.target_image_hight, config.target_image_width, 1])

# Input Tensor Shape: [batch_size, 128, 128, 1]
# Output Tensor Shape: [batch_size, 128, 128, 32]
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 128, 128, 32]
# Output Tensor Shape: [batch_size, 64, 64, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 64, 64, 32]
# Output Tensor Shape: [batch_size, 64, 64, 64]
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 64, 64, 64]
# Output Tensor Shape: [batch_size, 32, 32, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 32, 32, 64]
# Output Tensor Shape: [batch_size, 32, 32, 128]
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 32, 32, 128]
# Output Tensor Shape: [batch_size, 16, 16, 128]
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 16, 16, 128]
# Output Tensor Shape: [batch_size, 16, 16, 256]
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 16, 16, 256]
# Output Tensor Shape: [batch_size, 8, 8, 256]
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 8, 8, 256]
# Output Tensor Shape: [batch_size, 8, 8, 256]
conv5 = tf.layers.conv2d(
    inputs=pool4,
    filters=256,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 8, 8, 256]
# Output Tensor Shape: [batch_size, 4, 4, 256]
pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 4, 4, 256]
# Output Tensor Shape: [batch_size, 4, 4, 512]
conv6 = tf.layers.conv2d(
    inputs=pool5,
    filters=512,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

# Input Tensor Shape: [batch_size, 4, 4, 512]
# Output Tensor Shape: [batch_size, 2, 2, 512]
pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

# Input Tensor Shape: [batch_size, 2, 2, 512]
# Output Tensor Shape: [batch_size, 2 * 2 * 512]
pool6_flat = tf.reshape(pool6, [-1, 2 * 2 * 512])

dense1 = tf.layers.dense(inputs=pool6_flat, units=1024, activation=tf.nn.relu)
dropout1 = tf.layers.dropout(dense1, rate=0.25, training=config.train_MODE)

dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
dropout2 = tf.layers.dropout(dense2, rate=0.25, training=config.train_MODE)

out_layer = tf.layers.dense(inputs=dropout2, units=config.subset_label_array_length, activation=tf.nn.relu)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, epsilon=1)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    try:
        saver.restore(sess, weigth_path)
    except:
        print("no weight save found")
    dm = dataManager.DataManager()

    if (config.train_MODE):
        log.info("start training model 1...")
        batch_round = 1
        while True:
            batch_x, batch_y = dm.next_train_batch(config.train_batch_size)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            log.info("batch_round: {0}, cost in this train batch={1}".format(batch_round, cost))
            if np.mod(batch_round, 100) == 0:
                save_path = saver.save(sess, weigth_path)
                log.info('save model 1 weight')
            batch_round = batch_round + 1
    else:
        log.info("start testing model 1...")
        test_sample_batch_number = 0
        test_sample_total_accuracy = 0
        while (not dm.test_data_reach_end_flag):
            batch_x, batch_y = dm.next_test_batch(config.test_batch_size)
            pred = tf.nn.softmax(out_layer)
            correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
            predict_result, Y_result = sess.run([out_layer, Y], feed_dict={X: batch_x, Y: batch_y})
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_eval = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            log.info("model accuracy in this test batch: %f" % accuracy_eval)
            test_sample_batch_number = test_sample_batch_number + 1
            test_sample_total_accuracy = test_sample_total_accuracy + accuracy_eval
        test_sample_average_accuracy = test_sample_total_accuracy / test_sample_batch_number
        log.info("finished testing model 1.")
        log.info("average model accuracy of this test dataset: %f" % test_sample_average_accuracy)

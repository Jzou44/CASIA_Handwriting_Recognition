import tensorflow as tf
import numpy as np
import config, DataManager, Logger, NN_Model

log = Logger.get_logger('Model_3', 'log/Model_3.log')
weight_path = 'weight/Model_3/Model_3.ckpt'
character_amount = 5
each_character_sample_amount = 40


def calculate_variance(classification_layer):
    character_layer = tf.reshape(classification_layer,
                                 shape=[character_amount, each_character_sample_amount, config.label_array_length])
    character_layer_mean, character_layer_variance = tf.nn.moments(character_layer, [1])
    loss = tf.reduce_mean(character_layer_variance)
    return loss


# tf Graph input
X = tf.placeholder("float", [None, config.image_hight, config.image_width])
Y = tf.placeholder("float", [None, config.label_array_length])

feature_layer = NN_Model.cnn(X, Y)
classification_layer = NN_Model.full_connected_classifier(feature_layer)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1)

loss_variance = calculate_variance(classification_layer)
#
loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classification_layer, labels=Y))
loss_softmax_plus_variance = tf.add(loss_softmax, tf.multiply(0.1, loss_variance))
train_op = optimizer.minimize(loss_softmax_plus_variance)
# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    try:
        saver.restore(sess, weight_path)
    except:
        log.info("no saved weight found.")

    dm = DataManager.DataManager()
    if (config.MODE == tf.estimator.ModeKeys.TRAIN):
        log.info("start training model...")
        batch_round = 1
        while True:
            batch_x, batch_y = dm.next_train_batch_fix_character_amount(character_amount, each_character_sample_amount)
            _, cost, cost_s, cost_v = sess.run([train_op, loss_softmax_plus_variance, loss_softmax, loss_variance],
                                               feed_dict={X: batch_x, Y: batch_y})
            log.info("batch_round: {0},cost={1},softmax={2},loss_variance={3}".format(batch_round, cost, cost_s, cost_v))
            if np.mod(batch_round, 100) == 0:
                save_path = saver.save(sess, weight_path)
                log.info('save weight')
            batch_round = batch_round + 1

    if (config.MODE == tf.estimator.ModeKeys.EVAL):
        log.info("start evaluate model...")
        test_sample_batch_number = 0
        test_sample_total_accuracy = 0
        while True:
            batch_x, batch_y = dm.next_test_batch(200)
            if (len(batch_x) == 0):
                break
            pred = tf.nn.softmax(classification_layer)
            correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_eval = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            log.info("model accuracy in this test batch: %f" % accuracy_eval)
            test_sample_batch_number = test_sample_batch_number + 1
            test_sample_total_accuracy = test_sample_total_accuracy + accuracy_eval
        test_sample_average_accuracy = test_sample_total_accuracy / test_sample_batch_number
        log.info("finished testing model.")
        log.info("average model accuracy of this test dataset: %f" % test_sample_average_accuracy)

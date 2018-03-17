import tensorflow as tf
import numpy as np
import config, DataManager, Logger, NN_Model

log = Logger.get_logger('Model_2', 'log/Model_2.log')
weight_path = 'weight/Model_2/Model_2.ckpt'
character_amount = 90
each_character_sample_amount = 2


def calculate_euclidean(classification_layer):
    character_layer_1 = tf.strided_slice(classification_layer, begin=[0, 0], end=[200 * 2, config.label_array_length],
                                         strides=[2, 1])
    character_layer_2 = tf.strided_slice(classification_layer, begin=[1, 0], end=[200 * 2, config.label_array_length],
                                         strides=[2, 1])
    loss = tf.pow(tf.nn.l2_loss(character_layer_1 - character_layer_2), 0.5)
    return loss


# tf Graph input
X = tf.placeholder("float", [None, config.image_hight, config.image_width])
Y = tf.placeholder("float", [None, config.label_array_length])

feature_layer = NN_Model.cnn(X, Y)
classification_layer = NN_Model.full_connected_classifier(feature_layer)

loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classification_layer, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1)
train_op = optimizer.minimize(loss_softmax)

loss_euclidean = calculate_euclidean(classification_layer)
#
loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classification_layer, labels=Y))
loss_softmax_plus_euclidean = tf.add(loss_softmax, tf.multiply(0.05, loss_euclidean))

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
            batch_x, batch_y = dm.next_train_batch_random_select(200)
            _, cost, cost_s, c_e = sess.run([train_op, loss_softmax_plus_euclidean, loss_softmax, loss_euclidean],
                                            feed_dict={X: batch_x, Y: batch_y})
            log.info("batch_round: {0},cost={1},softmax={2},euclidean={3}".format(batch_round, cost, cost_s, c_e))
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

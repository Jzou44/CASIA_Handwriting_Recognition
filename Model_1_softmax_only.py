import tensorflow as tf
import numpy as np
import config, DataManager, Logger, NN_Model

log = Logger.get_logger('Model_1', 'log/Model_1.log')
weight_path = 'weight/Model_1/Model_1.ckpt'

# tf Graph input
X = tf.placeholder("float", [None, config.image_hight, config.image_width])
Y = tf.placeholder("float", [None, config.label_array_length])

out_layer = NN_Model.CNN(X, Y)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, epsilon=1)
train_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    try:
        saver.restore(sess, weight_path)
    except:
        print("no saved weight found.")

    dm = DataManager.DataManager()
    if (config.MODE == tf.estimator.ModeKeys.TRAIN):
        log.info("start training model...")
        batch_round = 1
        while True:
            batch_x, batch_y = dm.next_train_batch_random_select(300)
            _, cost = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
            log.info("batch_round: {0}, cost in this train batch={1}".format(batch_round, cost))
            if np.mod(batch_round, 100) == 0:
                save_path = saver.save(sess, weight_path)
                log.info('save weight')
            batch_round = batch_round + 1

    if (config.MODE == tf.estimator.ModeKeys.EVAL):
        log.info("start evaluate model...")
        test_sample_batch_number = 0
        test_sample_total_accuracy = 0
        while True:
            batch_x, batch_y = dm.next_test_batch(300)
            if (len(batch_x) == 0):
                break
            pred = tf.nn.softmax(out_layer)
            correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_eval = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            log.info("model accuracy in this test batch: %f" % accuracy_eval)
            test_sample_batch_number = test_sample_batch_number + 1
            test_sample_total_accuracy = test_sample_total_accuracy + accuracy_eval
        test_sample_average_accuracy = test_sample_total_accuracy / test_sample_batch_number
        log.info("finished testing model.")
        log.info("average model accuracy of this test dataset: %f" % test_sample_average_accuracy)

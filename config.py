import tensorflow as tf

CASIA_train_sqlite_file_path = 'Dataset/TrainDataset.sqlite'
CASIA_test_sqlite_file_path = 'Dataset/TestDataset.sqlite'
CASIA_label_file_path = 'Dataset/label.pickle'

image_hight = 128
image_width = 128
label_array_length = 300
learning_rate = 0.001

MODE = tf.estimator.ModeKeys.TRAIN

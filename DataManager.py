import numpy as np
import tensorflow as tf
import pickle, sqlite3
import config


class DataManager:
    def __init__(self):
        self.label_array = pickle.load(open(config.CASIA_label_file_path, 'rb'))  # train label array length = 300
        self.init_dataset()

    def init_dataset(self):
        if (config.MODE == tf.estimator.ModeKeys.TRAIN):
            self.CASIA_train_sqlite_connection = sqlite3.connect(config.CASIA_train_sqlite_file_path)
            self.CASIA_train_sqlite_cursor = self.CASIA_train_sqlite_connection.cursor()
            # save sample index into array
            self.CASIA_train_sqlite_cursor.execute("SELECT ID,Character_in_gb2312 FROM TrainDataset")
            train_dataset_index = self.CASIA_train_sqlite_cursor.fetchall()
            self.train_dataset_id_array = []
            self.train_dataset_character_dict = dict()
            for ID, Character_in_gb2312 in train_dataset_index:
                self.train_dataset_id_array.append(ID)
                if Character_in_gb2312 not in self.train_dataset_character_dict:
                    self.train_dataset_character_dict[Character_in_gb2312] = [ID]
                else:
                    self.train_dataset_character_dict[Character_in_gb2312].append(ID)
        if (config.MODE == tf.estimator.ModeKeys.EVAL):
            self.CASIA_test_sqlite_connection = sqlite3.connect(config.CASIA_test_sqlite_file_path)
            self.CASIA_test_sqlite_cursor = self.CASIA_test_sqlite_connection.cursor()
            self.CASIA_test_sqlite_cursor.execute("SELECT ID,Character_in_gb2312 FROM TestDataset")
            test_dataset_index = self.CASIA_test_sqlite_cursor.fetchall()
            self.test_dataset_id_array = []
            for ID, Character_in_gb2312 in test_dataset_index:
                self.test_dataset_id_array.append(ID)

    # ramdom select certain amount of training data
    def next_train_batch_random_select(self, batch_size=200):
        select_ids = np.random.choice(self.train_dataset_id_array, batch_size, replace=False)
        select_sql = "SELECT * FROM TrainDataset WHERE ID IN " + str(tuple(select_ids))
        self.CASIA_train_sqlite_cursor.execute(select_sql)
        select_data = self.CASIA_train_sqlite_cursor.fetchall()
        feature_batch, target_batch = self.process_sqlite_data(select_data)
        return feature_batch, target_batch

    # ramdom select x sample in each y classes
    def next_train_batch_fix_character_amount(self, character_amount=90, each_character_sample_amount=2):
        select_characters = np.random.choice(self.label_array, character_amount, replace=False)
        select_ids = []
        for character in select_characters:
            select_id = np.random.choice(self.train_dataset_character_dict[character], each_character_sample_amount,
                                         replace=False)
            select_ids.extend(select_id)
        select_sql = "SELECT * FROM TrainDataset WHERE ID IN " + str(
            tuple(select_ids)) + " ORDER BY Character_in_gb2312"
        self.CASIA_train_sqlite_cursor.execute(select_sql)
        select_data = self.CASIA_train_sqlite_cursor.fetchall()
        feature_batch, target_batch = self.process_sqlite_data(select_data)
        return feature_batch, target_batch

    def next_test_batch(self, batch_size=200):
        if len(self.test_dataset_id_array) >= batch_size:
            select_ids = np.random.choice(self.test_dataset_id_array, batch_size, replace=False)
        else:
            select_ids = self.test_dataset_id_array
        self.test_dataset_id_array = list(set(self.test_dataset_id_array) - set(select_ids))
        select_sql = "SELECT * FROM TestDataset WHERE ID IN " + str(tuple(select_ids))
        self.CASIA_test_sqlite_cursor.execute(select_sql)
        select_data = self.CASIA_test_sqlite_cursor.fetchall()
        feature_batch, target_batch = self.process_sqlite_data(select_data)
        return feature_batch, target_batch

    def process_sqlite_data(self, sqlite_data):
        feature_batch = np.zeros(shape=(len(sqlite_data), config.image_hight, config.image_width), dtype=np.float32)
        target_batch = np.zeros(shape=(len(sqlite_data), len(self.label_array)), dtype=np.float32)
        for i in range(len(sqlite_data)):
            # construct features
            img = pickle.loads(sqlite_data[i][5])
            height, width = np.shape(img)
            # 1. crop by (target_image_hight * target_image_width)
            if height > config.image_hight:
                img = img[(height - config.image_hight) // 2:(height + config.image_hight) // 2, :]
            if width > config.image_width:
                img = img[:, (width - config.image_width) // 2:(width + config.image_width) // 2]
            # pad to (target_image_hight * target_image_width)
            height, width = np.shape(img)
            top_pad = (config.image_hight - height) // 2
            bottom_pad = (config.image_hight - height) // 2 + np.mod(config.image_hight - height, 2)
            left_pad = (config.image_width - width) // 2
            right_pad = (config.image_width - width) // 2 + np.mod(config.image_width - width, 2)
            img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant', constant_values=255)
            # 3. scale graylevel 255. to 0. and graylevel 0. to 1.
            img = (255 - img)
            img = img.astype(np.float32)
            feature_batch[i, :, :] = img
            # construct target
            label = [int(x == sqlite_data[i][1]) for x in self.label_array]
            target_batch[i, :] = label
        return feature_batch, target_batch
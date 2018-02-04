import numpy as np
import pickle
import config as config


class DataManager:
    def __init__(self):
        self.CASIA_train_file = open(config.CASIA_train_file_path, 'rb')
        self.CASIA_test_file = open(config.CASIA_test_file_path, 'rb')
        self.CASIA_label_file = open(config.CASIA_label_file_path, 'rb')
        self.full_label_array = pickle.load(self.CASIA_label_file)  # train label array length = 3740
        self.subset_label_array = self.full_label_array[:config.subset_label_array_length]
        self.test_data_reach_end_flag = False

    def next_train_batch(self, batch_size):
        return self.next_batch(batch_size, self.next_train_sample)

    def next_test_batch(self, batch_size):
        return self.next_batch(batch_size, self.next_test_sample)

    def next_batch(self, batch_size, sample_function):
        # get batch_size number of sample every time
        image_matrix_sample_batch = np.zeros(shape=(batch_size, config.target_image_hight, config.target_image_width),
                                             dtype=np.float32)
        label_sample_batch = np.zeros(shape=(batch_size, config.subset_label_array_length), dtype=np.float32)
        for i in range(batch_size):
            image_matrix_sample, label_sample = sample_function()
            image_matrix_sample_batch[i, :, :] = image_matrix_sample
            label_sample_batch[i, :] = label_sample
        return [image_matrix_sample_batch, label_sample_batch]

    def next_train_sample(self):
        return self.next_sample_from_file(self.CASIA_train_file)

    def next_test_sample(self):
        return self.next_sample_from_file(self.CASIA_test_file)

    def next_sample_from_file(self, file):
        while True:
            # read header, header contain chinese_character_in_integer,width,height
            header = np.fromfile(file, dtype=np.uint8, count=config.dataset_header_size)
            if (header.size < config.dataset_header_size):
                file.seek(0)
                if (file == self.CASIA_test_file): self.test_data_reach_end_flag = True
                continue
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            chinese_character_in_integer = (header[4] << 8) + header[5]
            width = (header[7] << 8) + header[6]
            height = (header[9] << 8) + header[8]
            # image is next to header
            image_matrix = np.fromfile(file, dtype=np.uint8, count=width * height).reshape(height, width)
            # if this label belongs to subset, return this label
            if chinese_character_in_integer in self.subset_label_array:
                image_matrix_sample = self.preprocess_image(image_matrix)
                label_sample = [int(x == chinese_character_in_integer) for x in self.subset_label_array]
                return [image_matrix_sample, label_sample]
            else:
                continue

    def preprocess_image(self, img):
        height, width = np.shape(img)
        # 1. crop by (target_image_hight * target_image_width)
        if height > config.target_image_hight:
            img = img[(height - config.target_image_hight) // 2:(height + config.target_image_hight) // 2, :]
        if width > config.target_image_width:
            img = img[:, (width - config.target_image_width) // 2:(width + config.target_image_width) // 2]
        # pad to (target_image_hight * target_image_width)
        height, width = np.shape(img)
        top_pad = (config.target_image_hight - height) // 2
        bottom_pad = (config.target_image_hight - height) // 2 + np.mod(config.target_image_hight - height, 2)
        left_pad = (config.target_image_width - width) // 2
        right_pad = (config.target_image_width - width) // 2 + np.mod(config.target_image_width - width, 2)
        img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant', constant_values=255)
        # 3. scale graylevel 255. to 0. and graylevel 0. to 255.
        image = (255 - img)
        image = image.astype(np.float32)
        return image

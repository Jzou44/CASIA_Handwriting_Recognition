# CASIA_Handwriting_Recognition
course project for ECE 9603B Data Analytics Foundations

Blog: https://coding.tools/blog/casia-handwritten-chinese-character-recognition-using-convolutional-neural-network-and-similarity-ranking

## Usage

Get dataset from CASIA, details in http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

Train dataset download url: http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0train-gb1.rar

Test dataset download url: http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0test-gb1.rar

Convert dataset file as 'TrainDataset.sqlite','TestDataset.sqlite', save to Dataset/

Set Mode in 'config.py' (Train/Eval)

run model with command 'python Model_1_softmax_only.py'

3 different model is provided.

Model_1_softmax_only accuracy: 0.937968

Model_2_softmax_plus_Euclidean accuracy: 0.949909

Model_3_softmax_plus_Variance accuracy: 0.958333

Model_3_softmax_plus_Variance is designed by myself.

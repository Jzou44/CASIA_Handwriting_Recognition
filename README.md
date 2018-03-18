# CASIA_Handwriting_Recognition
course project for ECE 9603B Data Analytics Foundations


###Usage

Get dataset from CASIA, details in http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

Train dataset download url: http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0train-gb1.rar

Test dataset download url: http://www.nlpr.ia.ac.cn/databases/Download/feature_data/1.0test-gb1.rar

Convert dataset file as 'TrainDataset.sqlite','TestDataset.sqlite', save to Dataset/
Set Mode in 'config.py'
run mode with command 'python Model_1_softmax_only.py'
3 differnt model is provided.

Model_3_softmax_plus_Variance is designed by myself.
import numpy as np

from pyspark import SparkConf, SparkContext
from models import neural_networks, logistic_regression, SVM, kNN

#############################################################
#        Neural Networks, Logistic Regression, SVM          #
#############################################################

NN_weights = []
NN_biases = []

for i in range(1, 4):
    NN_weights.append(np.load("../keras/models/w_{}.npy".format(i)))
    NN_biases.append(np.load("../keras/models/b_{}.npy".format(i)))

def loaded_neural_networks(x):
	""" 
	function:
	used for "map" function in spark,
	whidh only takes 1 argument as input.

	weights already in global variables (NN_weights, NN_biases)
	"""
	return neural_networks(x, NN_weights, NN_biases)

LR_w = np.load("../sklearn/model_LR/w.npy")
LR_b = np.load("../sklearn/model_LR/b.npy")

def loaded_logistic_regression(x):
	"""
	function:
	used for "map" function in spark,
	whidh only takes 1 argument as input.

	weights already in global variables (LR_w, LR_b)
	"""
	return logistic_regression(x, LR_w, LR_b)

SVM_w = np.load("../sklearn/model_SVM/w.npy")
SVM_b = np.load("../sklearn/model_SVM/b.npy")

def loaded_SVM(x):
	"""
	function:
	used for "map" function in spark,
	whidh only takes 1 argument as input.

	weights already in global variables (SVM_w, SVM_b)
	"""
	return SVM(x, SVM_w, SVM_b)

#############################################################
#                               KNN                         #
#############################################################

kNN_k = 5
kNN_X_train = np.load("../data/subsamp_data/processed_X_train.npy")
kNN_y_train = np.load("../data/subsamp_data/processed_y_train.npy")

def loaded_kNN(x):
	"""
	function:
	used for "map" function in spark,
	whidh only takes 1 argument as input.

	weights already in global variables (kNN_k, kNN_X_train, kNN_y_train)
	"""

	return kNN(kNN_X_train, x, kNN_y_train, kNN_k)



if __name__ == "__main__":

	sc = SparkContext()

	test_data = np.load("../data/origin_data/X_test.npy")
	test_lable = np.load("../data/origin_data/y_test.npy")

	test_data_tuples = []
	for i in range(test_data.shape[0]):
		test_data_tuples.append((i, test_data[i]))

	print()

	x = sc.parallelize(test_data_tuples)

	NN_result = x.map(loaded_neural_networks).collect()
	LR_result = x.map(loaded_logistic_regression).collect()
	SVM_result = x.map(loaded_SVM).collect()
	# kNN_result = x.map(loaded_kNN).collect()

	print(NN_result[0:10])
	print(LR_result[0:10])
	print(SVM_result[0:10])
	# print(kNN_result[0:10])






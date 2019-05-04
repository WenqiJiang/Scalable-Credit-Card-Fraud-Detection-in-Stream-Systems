import numpy as np
import time

from pyspark import SparkConf, SparkContext
from models import neural_networks, logistic_regression, SVM, kNN

#############################################################
#        Neural Networks, Logistic Regression, SVM          #
#############################################################

NN_weights = []
NN_biases = []

for i in range(1, 4):
    NN_weights.append(np.load("../keras/models_best/w_{}.npy".format(i)))
    NN_biases.append(np.load("../keras/models_best/b_{}.npy".format(i)))

def loaded_neural_networks(x):
	""" 
	function:
	used for "map" function in spark,
	whidh only takes 1 argument as input.

	weights already in global variables (NN_weights, NN_biases)
	"""
	return neural_networks(x, NN_weights, NN_biases)

LR_w = np.load("../sklearn/model_LR/LR_best/w.npy")
LR_b = np.load("../sklearn/model_LR/LR_best/b.npy")

def loaded_logistic_regression(x):
	"""
	function:
	used for "map" function in spark,
	whidh only takes 1 argument as input.

	weights already in global variables (LR_w, LR_b)
	"""
	return logistic_regression(x, LR_w, LR_b)

SVM_w = np.load("../sklearn/model_SVM/SVM_best/w.npy")
SVM_b = np.load("../sklearn/model_SVM/SVM_best/b.npy")

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

	conf = SparkConf().setMaster("local[1]")
	sc = SparkContext(conf=conf)

	test_data = np.load("../data/origin_data/X_test.npy")
	test_lable = np.load("../data/origin_data/y_test.npy")

	test_data_tuples = []
	for i in range(test_data.shape[0]):
		test_data_tuples.append((i, test_data[i]))

	x = sc.parallelize(test_data_tuples)

	start_NN = time.perf_counter()
	NN_result = x.map(loaded_neural_networks).collect()
	end_NN = time.perf_counter()
	profiling_NN = end_NN - start_NN

	start_LR = time.perf_counter()
	LR_result = x.map(loaded_logistic_regression).collect()
	end_LR = time.perf_counter()
	profiling_LR = end_LR - start_LR

	start_SVM = time.perf_counter()
	SVM_result = x.map(loaded_SVM).collect()
	end_SVM = time.perf_counter()
	profiling_SVM = end_SVM - start_SVM

	start_kNN = time.perf_counter()
	kNN_result = sc.parallelize(test_data_tuples[:100]).map(loaded_kNN).collect()
	end_kNN = time.perf_counter()
	profiling_kNN = (end_kNN - start_kNN) * test_data.shape[0] / 100

	print(NN_result[0:10])
	print(LR_result[0:10])
	print(SVM_result[0:10])
	print(kNN_result[0:10])
	print("Validation data number: {}".format(test_data.shape[0]))
	print("NN time:\t{}\tLR time:\t{}\tSVM time:\t{}\tkNN time: \t{}\t".format(profiling_NN, profiling_LR, profiling_SVM, profiling_kNN))
	# print(kNN_result[0:10])

import numpy as np
import time
import argparse

from pyspark import SparkConf, SparkContext
from models import neural_networks, logistic_regression, SVM, kNN, \
					affine_forward, relu_forward, sigmoid
from sklearn.ensemble import RandomForestClassifier

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
	which only takes 1 argument as input.

	weights already in global variables (NN_weights, NN_biases)
	"""
	return neural_networks(x, NN_weights, NN_biases)

LR_w = np.load("../sklearn/model_LR/LR_best/w.npy")
LR_b = np.load("../sklearn/model_LR/LR_best/b.npy")

def loaded_logistic_regression(x):
	"""
	function:
	used for "map" function in spark,
	which only takes 1 argument as input.

	weights already in global variables (LR_w, LR_b)
	"""
	return logistic_regression(x, LR_w, LR_b)

SVM_w = np.load("../sklearn/model_SVM/SVM_best/w.npy")
SVM_b = np.load("../sklearn/model_SVM/SVM_best/b.npy")

def loaded_SVM(x):
	"""
	function:
	used for "map" function in spark,
	which only takes 1 argument as input.

	weights already in global variables (SVM_w, SVM_b)
	"""
	return SVM(x, SVM_w, SVM_b)

#############################################################
#                               KNN                         #
#############################################################

kNN_k = 5
kNN_X_train = np.load("../data/origin_data/X_train.npy")
kNN_y_train = np.load("../data/origin_data/y_train.npy")

def loaded_kNN(x):
	"""
	function:
	used for "map" function in spark,
	which only takes 1 argument as input.

	weights already in global variables (kNN_k, kNN_X_train, kNN_y_train)
	"""

	return kNN(kNN_X_train, x, kNN_y_train, kNN_k)

# RF_clf = RandomForestClassifier(n_estimators=1, max_depth=5,
# 				random_state=0,class_weight = {0:1,1:1})
# RF_clf.fit(kNN_X_train, kNN_y_train)

# def loaded_RandomForest(x):
# 	"""
# 	function:
# 	used for "map" function in spark,
# 	which only takes 1 argument as input.

# 	weights already in global variables RF_clf
# 	"""
# 	return RF_clf.predict(x)

def NN_pipeline1(x):
    assert len(NN_weights) == len(NN_biases)
    
    # extract feature maps in tuples
    if type(x) == type((1,2)):
        input_FM = x[1]
        weight = NN_weights[0]
        bias = NN_biases[0]
        output_FM = affine_forward(input_FM, weight, bias)
        output_FM = relu_forward(output_FM)

        return (x[0], output_FM)

    else:
        input_FM = []
        # print(len(x))
        for i in range(len(x)):
            input_FM.append(x[i][1])
        input_FM = np.reshape(np.array(input_FM), (len(x), -1))
    
        weight = NN_weights[0]
        bias = NN_biases[0]
        output_FM = affine_forward(input_FM, weight, bias)
        output_FM = relu_forward(output_FM)		

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], output_FM[i]))

        return result

def NN_pipeline2(x):
    assert len(NN_weights) == len(NN_biases)
    
    # extract feature maps in tuples
    if type(x) == type((1,2)):
        input_FM = x[1]
        weight = NN_weights[1]
        bias = NN_biases[1]
        output_FM = affine_forward(input_FM, weight, bias)
        output_FM = relu_forward(output_FM)

        return (x[0], output_FM)

    else:
        input_FM = []
        # print(len(x))
        for i in range(len(x)):
            input_FM.append(x[i][1])
        input_FM = np.reshape(np.array(input_FM), (len(x), -1))
    
        weight = NN_weights[1]
        bias = NN_biases[1]
        output_FM = affine_forward(input_FM, weight, bias)
        output_FM = relu_forward(output_FM)		

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], output_FM[i]))

        return result

def NN_pipeline3(x):
    """
    x: tuples, (index, (features))
    NN_weights: list of NN_weights, [W1, W2, W3 ...]
    NN_biases: list of NN_biases, [b1, b2, b3 ...]
    
    return:
    the result lable
    """
    assert len(NN_weights) == len(NN_biases)
    
    # extract feature maps in tuples
    if type(x) == type((1,2)):
        input_FM = x[1]
        weight = NN_weights[2]
        bias = NN_biases[2]
        output_FM = affine_forward(input_FM, weight, bias)
        
        # if we don't want to know the probability, we can remove sigmoid
        prob = sigmoid(output_FM)
        if prob >= 0.5:
            result = (x[0], int(1))
        else:
            result = (x[0], int(0))

    else:
        input_FM = []
        # print(len(x))
        for i in range(len(x)):
            input_FM.append(x[i][1])
        input_FM = np.reshape(np.array(input_FM), (len(x), -1))
    
        weight = NN_weights[2]
        bias = NN_biases[2]
        output_FM = affine_forward(input_FM, weight, bias)
        
        # if we don't want to know the probability, we can remove sigmoid
        prob = sigmoid(output_FM)
        result_arr = np.zeros(prob.shape, dtype=np.int32)
        #     print("aaa", prob)
        result_arr[np.where(prob >= 0.5)] = 1

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], result_arr[i]))

    return result

def algorithms_wrapper(x):
	"""
	Wrap up three algorithms we will use:
	for different speed range we will pick different algorithms,	
	from 0 to 11,873 sample/s, we choose Neural Network      		
	from 11,873 to 38,141 sample/s, we use Logistic Regression      
	from 38,141 to 45,080 samples / s, we pick SVM 				    
	for faster speed, out of our ability to process data 			
	"""
	return None


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=1, help="batch size, default 20")
	######################################################################
	#						option mode description 					 #
	# profiling: run 10 iterations and profile average time				 #
	# normal: run SVM, LR and NN 										 #
	# dynamic: dynamically pick the algorithm given input speed 		 #
	######################################################################
	parser.add_argument('--option', type=str, default="normal", help="choices: profiling / dynamic / normal")
	parser.add_argument('--dataset', type=str, default="subsample", help="choices: subsample / origin")
	args = parser.parse_args()
	batch_size = args.batch_size
	option = args.option
	dataset = args.dataset

	conf = SparkConf().setMaster("local[1]")
	sc = SparkContext(conf=conf)
	sc.setLogLevel('OFF')

	if dataset == "subsample":
		test_data = np.load("../data/subsamp_data/processed_X_test.npy")
		test_lable = np.load("../data/subsamp_data/processed_y_test.npy")
	if dataset == "origin":
		# raise Exception ("Almost all result of origin dataset are 0s, please use subsampled dataset for demo")
		test_data = np.load("../data/origin_data/X_test.npy")
		test_lable = np.load("../data/origin_data/y_test.npy")

	test_data_tuples = []
	for i in range(test_data.shape[0]):
		test_data_tuples.append((i, test_data[i]))

	data_num = len(test_data_tuples)
	if batch_size > data_num:
		raise Exception("Invalid batch size: larger than the whole dataset")
	batchs = data_num // batch_size
	test_data_tuples = [ test_data_tuples[i * batch_size: (i + 1) * batch_size] for i in range(batchs)]
	if batchs * batch_size < data_num:
		test_data_tuples_tail = test_data_tuples[batchs * batch_size:]
		if test_data_tuples_tail != []:
			test_data_tuples.append(test_data_tuples_tail)

	if option == "profiling":

		x = sc.parallelize(test_data_tuples)
		iter_num = 10
		profiling_NN = 0
		profiling_LR = 0
		profiling_SVM = 0
		profiling_pipeline_NN = 0
		for i in range(iter_num):
			start_NN = time.perf_counter()
			NN_result = x.map(loaded_neural_networks).collect()
			end_NN = time.perf_counter()
			profiling_NN += end_NN - start_NN

			start_LR = time.perf_counter()
			LR_result = x.map(loaded_logistic_regression).collect()
			end_LR = time.perf_counter()
			profiling_LR += end_LR - start_LR

			start_SVM = time.perf_counter()
			SVM_result = x.map(loaded_SVM).collect()
			end_SVM = time.perf_counter()
			profiling_SVM += end_SVM - start_SVM

			start_pipeline_NN = time.perf_counter()
			pipeline_NN_result = x.map(NN_pipeline1).map(NN_pipeline2).map(NN_pipeline3).collect()
			end_pipeline_NN = time.perf_counter()
			profiling_pipeline_NN += end_pipeline_NN - start_pipeline_NN

		print(NN_result[0:10])
		print(LR_result[0:10])
		print(SVM_result[0:10])
		print(pipeline_NN_result[0:10])
		# print(kNN_result[0:10])
		print("Validation data number: {}".format(test_data.shape[0]))
		print("NN time:\t{}\tLR time:\t{}\tSVM time:\t{}\t".format(profiling_NN / iter_num, 
									profiling_LR / iter_num, profiling_SVM / iter_num))
		print("pipeline NN time:\t{}\t".format(profiling_pipeline_NN / 10))

	elif option == "normal":
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

		start_pipeline_NN = time.perf_counter()
		pipeline_NN_result = x.map(NN_pipeline1).map(NN_pipeline2).map(NN_pipeline3).collect()
		end_pipeline_NN = time.perf_counter()
		profiling_pipeline_NN = end_pipeline_NN - start_pipeline_NN

		print(NN_result[0:10])
		print(LR_result[0:10])
		print(SVM_result[0:10])
		print(pipeline_NN_result[0:10])
		# print(kNN_result[0:10])
		print("Validation data number: {}".format(test_data.shape[0]))
		print("NN time:\t{}\tLR time:\t{}\tSVM time:\t{}\tpipeline NN time:\t{}\t".format(profiling_NN, 
			profiling_LR, profiling_SVM, profiling_pipeline_NN))

	elif option == "dynamic":
		#####################################################################
		#	for different speed range we will pick different algorithms,	#
		#	from 0 to 11,873 sample/s, we choose Neural Network      		#
		#	from 11,873 to 38,141 sample/s, we use Logistic Regression      #
		#	from 38,141 to 45,080 samples / s, we pick SVM 				    #
		#	for faster speed, out of our ability to process data 			#
		#####################################################################
		data_num = len(test_data_tuples) * batch_size
		loop_time = 1 # each iteration last for 0.1 seconds

		for i in range(300):
			start = time.perf_counter()
			speed = np.random.randint(1000, 45080)
			# speed * 0.1 -> data we will process
			batch_num = int((speed * loop_time) // batch_size)
			x = sc.parallelize(test_data_tuples[:batch_num])

			# NN 
			if speed <= 11873:
				algorithm = "Neural Network"
				result = x.map(loaded_neural_networks).collect()
			
			# LR
			elif speed > 11873 and speed <= 38141:
				algorithm = "Logistic Regression"
				result = x.map(loaded_logistic_regression).collect()

			# SVM
			elif speed > 38142 and speed <= 45080:
				algorithm = "Support Vector Machine"
				result = x.map(loaded_SVM).collect()

			# speed too high
			else:
				algorithm = None
				result = None
				raise Exception("Given streaming speed out of computing capacity")

			print_len = 5
			sample_result = []
			for i in range(batchs):
				sample_result += result[i]
			start_idx = np.random.randint(0, len(sample_result) - print_len - 2)
			sample_result = sample_result[start_idx: start_idx + print_len]

			print("Data feed in speed: {}\tAlgorithm used:\t{}".format(speed, algorithm))
			print("Sample predictions:\t{}\n\n".format(sample_result))
			end = time.perf_counter()

			if end - start < loop_time:
				time.sleep(loop_time - (end - start))

	else:
		raise Exception("Unrecognized option, please use one of these options:\nnoraml, dynamic, profiling")

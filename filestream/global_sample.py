import numpy as np

from pyspark import SparkConf, SparkContext

y = 1
def my_func(x, y):
	"""
	wrong way
	"""
	return x + 1

def my_func2(x):
	"""
	correct way: put y in global
	"""
	return x + y

if __name__ == "__main__":

	sc = SparkContext()

	test_data = np.load("../data/origin_data/X_test.npy")
	test_lable = np.load("../data/origin_data/y_test.npy")

	x = sc.parallelize(test_data)


	result = x.map(my_func2).collect()

	print(test_data[0])
	print(result[0])
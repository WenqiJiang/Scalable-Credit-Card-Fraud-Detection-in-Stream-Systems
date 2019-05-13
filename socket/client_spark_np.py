from __future__ import print_function

import sys
import numpy as np
import argparse

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]
    
    return out

def sigmoid(x):
    """
    Sigmoid function, vectorized version.
    
    :param x: (float) a tensor of shape (N, #classes)
    """
    return (1 / (1 + np.exp(-x)))


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    """
    ## let me hardcoding yibo :)
    if x.shape[0] == w.shape[0]:
        x = np.reshape(x, (1, x.shape[0]))

    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    
    return out

def neural_networks(x, NN_weights, NN_biases):
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
        for i in range(len(NN_weights)):
            weight = NN_weights[i]
            bias = NN_biases[i]
            output_FM = affine_forward(input_FM, weight, bias)
            if i != len(NN_weights) - 1:
                output_FM = relu_forward(output_FM)
            input_FM = output_FM
        
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
    
    
        for i in range(len(NN_weights)):
            weight = NN_weights[i]
            bias = NN_biases[i]
            output_FM = affine_forward(input_FM, weight, bias)
            if i != len(NN_weights) - 1:
                output_FM = relu_forward(output_FM)
            input_FM = output_FM
        
        # if we don't want to know the probability, we can remove sigmoid
        prob = sigmoid(output_FM)
        result_arr = np.zeros(prob.shape, dtype=np.int32)
        #     print("aaa", prob)
        result_arr[np.where(prob >= 0.5)] = 1

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], result_arr[i][0]))

    return result

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

def np_return(x):
    '''
    Return feed_freq (int), batch_size (int), list of a batch of transactions [(index (int), transaction(numpy.array))]
    '''
    # arr = np.frombuffer(x.encode())
    # return arr
    # return x[1:-1].split()
    # return np.array(x[1:-1].split(), dtype=np.float64)
    # print('size:    ',sys.getsizeof(x.encode()))
    x_split = x.split()
    # return int(x_split[0]), int(x_split[1]), np.array(x_split[2:], dtype=np.float64)
    feed_freq = int(x_split[0])
    batch_size = int(x_split[1])

    x_split = x_split[2:]
    feedin_data = []
    for i in range(batch_size):
        index = int(x_split[i * 29])
        feature = np.array(x_split[i*29 + 1 : (i+1)*29], dtype=np.float64)
        feedin_data.append((index,feature))
    return feedin_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6666, help="network port")
    args = parser.parse_args()
    serverPort = args.port

    conf = SparkConf().setMaster("local[2]")
    sc = SparkContext(conf = conf,appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, 0.01)

    sc.setLogLevel('OFF')

    # lines = ssc.socketTextStream('localhost', 8888)
    lines = ssc.socketTextStream("localhost", serverPort)
    result = lines.map(np_return).map(loaded_neural_networks)

    result.pprint()

    ssc.start()
    ssc.awaitTermination()
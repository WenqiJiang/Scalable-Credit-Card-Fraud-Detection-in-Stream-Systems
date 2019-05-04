import numpy as np
import operator

#############################################################
#        Neural Networks, Logistic Regression, SVM          #
#############################################################

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

def softmax(x):
    """
    Softmax loss function, vectorized version.
    
    :param x: (float) a tensor of shape (N, #classes)
    """
    delta = np.max(x, axis=1)
    x -= delta
    exp_x = np.exp(x)
    sumup = np.sum(exp_x, axis=1)
    
    result = exp_x / sumup
    return result

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

def SVM(x, SVM_w, SVM_b):
    """
    x: tuples, (index, (features))
    SVM_w: weights
    SVM_b: bias
    
    return:
    the result lable
    """
    if type(x) == type((1,2)): # a single tuple
        input_FM = x[1]
        output_FM = affine_forward(input_FM, SVM_w, SVM_b)
        if output_FM > 0:
            result = (x[0], 1)
        else:
            result = (x[0], 0)
    else:   # a list of tuple
        # preprocess 
        input_FM = []
        for i in range(len(x)):
            input_FM.append(x[i][1])
        input_FM = np.array(input_FM)

        output_FM = affine_forward(input_FM, SVM_w, SVM_b)
        result_arr = np.zeros(input_FM.shape[0])
        
        # class 1: output_FM > 0:
        result_arr[np.where(output_FM >= 0)] = 1

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], result_arr[i]))

    return result


def logistic_regression(x, LR_w, LR_b):
    """
    x: tuples, (index, (features))
    LR_w: weights
    LR_b: LR_bias
    
    return:
    the result lable
    """
    if type(x) == type((1,2)): # a single tuple
        input_FM = x[1]
        output_FM = affine_forward(input_FM, LR_w, LR_b)
        prob = sigmoid(output_FM)
        if prob >= 0.5:
            result = (x[0], 1)
        else:
            result = (x[0], 0)
    else: # a list of tuples
        input_FM = []
        for i in range(len(x)):
            input_FM.append(x[i][1])
        input_FM = np.array(input_FM)

        output_FM = affine_forward(input_FM, LR_w, LR_b)
        
        # if we don't want to know the probability, we can remove softmax
        prob = sigmoid(output_FM)
        result_arr = np.zeros(prob.shape)
        result_arr[np.where(prob >= 0.5)] = 1

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], result_arr[i]))
    
    return result


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
            result = (x[0], 1)
        else:
            result = (x[0], 0)

    else:
        input_FM = []
        for i in range(len(x)):
            input_FM.append(x[i][1])
        input_FM = np.array(input_FM)
    
    
        for i in range(len(NN_weights)):
            weight = NN_weights[i]
            bias = NN_biases[i]
            output_FM = affine_forward(input_FM, weight, bias)
            if i != len(NN_weights) - 1:
                output_FM = relu_forward(output_FM)
            input_FM = output_FM
        
        # if we don't want to know the probability, we can remove sigmoid
        prob = sigmoid(output_FM)
        result_arr = np.zeros(prob.shape)
        #     print("aaa", prob)
        result_arr[np.where(prob >= 0.5)] = 1

        # process result
        result = []
        for i in range(len(x)):
            result.append((x[i][0], result_arr[i]))

    return result

#############################################################
#                               KNN                         #
#############################################################

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))

def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbors = []
    for i in range(X_train.shape[0]):
        dist = euclidean_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        #print distances[x]
        neighbors.append(distances[x][0])
    return neighbors

def predictkNNClass(neighbors, y_train):
    classVotes = {}
    for i in range(len(neighbors)):
        if y_train[neighbors[i]] in classVotes:
            classVotes[y_train[neighbors[i]]] += 1
        else:
            classVotes[y_train[neighbors[i]]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def kNN(X_train, X_test, Y_train, k):
    output_classes = []
    for i in range(X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes

import numpy as np

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
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    
    return out

def SVM(x, w, b):
    """
    x: (N, dim)
    w: weights
    b: bias
    
    return:
        the result lable
    """
    output_FM = affine_forward(x, w, b)
    result = np.zeros(x.shape[0])
    
    # class 1: output_FM > 0:
    result[np.where(x >= 0)] = 1
    
    return result

def logistic_regression(x, w, b):
    """
    x: (N, dim)
    w: weights
    b: bias
    
    return:
        the result lable
    """
    output_FM = affine_forward(x, w, b)
    
    # if we don't want to know the probability, we can remove softmax
    prob = sigmoid(output_FM)
    result = np.zeros(prob.shape)
    result[np.where(prob >= 0.5)] = 1
    
    return result

def neural_networks(x, weights, biases):
    """
    x: (N, dim)
    weights: list of weights, [W1, W2, W3 ...]
    biases: list of biases, [b1, b2, b3 ...]
    
    return:
        the result lable
    """
    input_FM = x
    assert len(weights) == len(biases)
    
    for i in range(len(weights)):
        weight = weights[i]
        bias = biases[i]
        output_FM = affine_forward(input_FM, weight, bias)
        if i != len(weights) - 1:
            output_FM = relu_forward(output_FM)
        input_FM = output_FM
        
    # if we don't want to know the probability, we can remove softmax
    prob = sigmoid(output_FM)
    result = np.zeros(prob.shape)
#     print("aaa", prob)
    result[np.where(prob >= 0.5)] = 1
    
    return result
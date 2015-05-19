##
# @brief 
# """
# This tutorial introduces logistic regression using Theano and stochastic
# gradient descent.
# 
# Logistic regression is a probabilistic, linear classifier. It is parametrized
# by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
# done by projecting data points onto a set of hyperplanes, the distance to
# which is used to determine a class membership probability.
# 
# Mathematically, this can be written as:
# 
# .. math::
#   P(Y=i|x, W,b) &= softmax_i(W x + b) \\
#                 &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}
# 
# 
# The output of the model or prediction is then done by taking the argmax of
# the vector whose i'th element is P(Y=i|x).
# 
# .. math::
# 
#   y_{pred} = argmax_i P(Y=i|x,W,b)
# 
# 
# This tutorial presents a stochastic gradient descent optimization method
# suitable for large datasets.
# 
# 
# References:
# 
#     - textbooks: "Pattern Recognition and Machine Learning" -
#                  Christopher M. Bishop, section 4.3.2
# 
# """
__docformat__ = 'restructedtext en'

import os
import sys
import numpy
import theano
import theano.tensor as T

import LossFunctions
import ModelUtility
import Classifier
import LogisticRegression
import rbm


##
# @brief 
#
# @param inputVariable
# @param n_in
# @param n_out
# @param layers
#
# @return 
def getMultilayerPerceptron(inputVariable, n_in, layers):
    # parameters of the model, will be used later to update
    # This can be stored and reload to replicate the same result
    params = {}
    output = inputVariable

    outputs = {}
    wVariables = {}
    bVariables = {}

    # The first output is the input variable
    outputs[0] = output

    # initialize each layer and put them into params 
    for i in range(len(layers)):
        # if this is first layer, use n_in as the input dimension,
        # otherwise use the previous layer number as input dimension
        if i == 0:
            inputNumber = n_in
        else:
            inputNumber = layers[i-1]
        outputNumber = layers[i]

        W = theano.shared(
            value=numpy.asarray(
                ModelUtility.getRandomNumpyMatrix(inputNumber, outputNumber),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # initialize the baises b as a vector of n_out 0s
        b = theano.shared(
            value=numpy.zeros(
                (outputNumber,),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        params["W" + str(i)] = W
        params["b" + str(i)] = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        output = T.tanh(T.dot(output, W) + b)

        # store output and parameters
        outputs[i + 1] = output
        wVariables[i + 1] = W
        bVariables[i + 1] = b


    return outputs, params, wVariables, bVariables
    

# Multi-Layer Perceptron Class
# 
# A multilayer perceptron is a feedforward artificial neural network model
# that has one layer or more of hidden units and nonlinear activations.
# Intermediate layers usually have as activation function tanh or the
# sigmoid function (defined here by a ``HiddenLayer`` class)  while the
# top layer is a softamx layer (defined here by a ``LogisticRegression``
# class).
# 
class MultilayerPerceptron(Classifier.Classifier):

    # must have property of learning model
    # this will used to upload to database
    description = "y_k = s(y_(k-1) * W + b), s is the activation function, repeat this process for k layers" 
    
    ##
    # @brief            Initialize the parameters 
    #
    # @param n_in       number of input units, the dimension of the space in
    #                       which the datapoints lie
    # @param n_out      number of output units, the dimension of the space in
    #                       which the labels lie
    # @param layers     an array of number that represent the number of neural in each layer 
    #
    # @return 
    def __init__(self, n_in, n_out, layers = [500]):
        # initialize classifier class
        super(MultilayerPerceptron, self).__init__()

        # variables to store layer info
        self.mlpOutputs, self.mlpParams, self.wVariables, self.bVariables = getMultilayerPerceptron(self._x, n_in, layers)


        logisticRegressionInputNumber = layers[-1]
        self.p_y_given_x, logisticRegressionParams= LogisticRegression.getLogisticRegressionLayer(self.mlpOutputs[len(layers)], logisticRegressionInputNumber, n_out) 

        self.params = {}
        for key in self.mlpParams:
            newKey = "mlp-" + key
            self.params[newKey] = self.mlpParams[key]
        for key in logisticRegressionParams:
            newKey = "logisticRegression-" + key
            self.params[newKey] = logisticRegressionParams[key]

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    def setPretrainLayer(self, layerNumber, batch_size, train_set_x, learning_rate = 0.1):
        
        self._pretrainModel =  rbm.getPretrainFunction(
                                       self.mlpOutputs[layerNumber-1], # the input variable is the previous layer's output 
                                       self._x, 
                                       self.wVariables[layerNumber], 
                                       self.bVariables[layerNumber], 
                                       batch_size, 
                                       train_set_x,
                                       learning_rate = learning_rate)

    def saveParameterAsImage(self, name):
        ModelUtility.saveMatrixAsImage(self.wVariables[1].get_value(), name)

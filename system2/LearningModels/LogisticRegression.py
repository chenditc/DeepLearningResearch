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


def getLogisticRegressionLayer(inputVariable, n_in, n_out, layerName = 'lgd'):
    # TODO: change initialize value to configurable, eg. random
    # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
    W = theano.shared(
        value=numpy.asarray(
            ModelUtility.getRandomNumpyMatrix(n_in, n_out),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    # initialize the baises b as a vector of n_out 0s
    b = theano.shared(
        value=numpy.zeros(
            (n_out,),
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    # parameters of the model, will be used later to update
    # This can be stored and reload to replicate the same result
    params = {
        layerName + '-' + 'W' : W, 
        layerName + '-' + 'b' : b
    }

    # symbolic expression for computing the matrix of class-membership
    # probabilities
    # Where:
    # W is a matrix where column-k represent the separation hyper plain for
    # class-k
    # x is a matrix where row-j  represents input training sample-j
    # b is a vector where element-k represent the free parameter of hyper
    # plain-k
    y_given_x = T.nnet.softmax(T.dot(inputVariable, params[layerName + '-' + 'W']) + params[layerName + '-' + 'b'])

    return y_given_x, params

##
# @brief 
# Multi-class Logistic Regression Class
# 
# The logistic regression is fully described by a weight matrix :math:`W`
# and bias vector :math:`b`. Classification is done by projecting data
# points onto a set of hyperplanes, the distance to which is used to
# determine a class membership probability.
# 
class LogisticRegression(Classifier.Classifier):

    # must have property of learning model
    # this will used to upload to database
    description = "y = x * W + b" 
    
    ##
    # @brief            Initialize the parameters of the logistic regression
    #
    # @param n_in       number of input units, the dimension of the space in
    #                       which the datapoints lie
    # @param n_out      number of output units, the dimension of the space in
    #                       which the labels lie
    # @param params     The parameters that can be used to initialize the model
    #
    # @return 
    def __init__(self, n_in, n_out):
        # initialize classifier class
        super(LogisticRegression, self).__init__()


        self.p_y_given_x, self.params= getLogisticRegressionLayer(self._x, n_in, n_out) 

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.buildTrainingModel()

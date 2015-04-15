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

    def __init__(self, n_in, n_out, inputVariable = None, outputVariable = None):
        """ Initialize the parameters of the logistic regression

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize classifier class
        super(LogisticRegression, self).__init__()

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        self._x = T.matrix('x')  # data, presented as rasterized images
        self._y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        if inputVariable != None:
            self._x = inputVariable
        if outputVariable != None:
            self._y = outputVariable

        # TODO: change initialize value to configurable, eg. random
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self._W = theano.shared(
            value=numpy.asarray(
                ModelUtility.getRandomNumpyMatrix(n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self._b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(self._x, self._W) + self._b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model, will be used later to update
        # This can be stored and reload to replicate the same result
        self.params = {
            'W' : self._W, 
            'b' : self._b
        }

        # initialize train model
        self._trainModel = None;
        self._testModel = theano.function(
            inputs=[self._x],
            outputs=self.y_pred,
        )


    ##
    # @brief                Create theano function that take training x and y
    #                       and update W and b
    #
    # @param train_set_x    
    # @param train_set_y
    # @param parameterToTrain   A list of parameter index to train in this training model
    # @param lossFunction       The loss function that will calculate gradient. A function takes x and y
    # @param learning_rate      the step updateing parameters
    # @param batch_size         The size of training data to compute at once
    #
    # @return 
    def buildTrainingModel(self, 
                           train_set_x, 
                           train_set_y,
                           parameterToTrain = [],
                           lossFunction = LossFunctions.LossFunctions.negative_log_likelihood, 
                           learning_rate = 0.1, 
                           batch_size = 600):

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        index = T.lscalar()  # index to a [mini]batch

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = lossFunction(self.p_y_given_x, self._y)

        updates = self.getUpdateForVariable(cost, learning_rate, self.params, onlyTrain=parameterToTrain)

        # compiling a Theano function `trainModel` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self._trainModel = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self._x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self._y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self._totalBatches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    ##
    # @brief                    Run the index to go through the training set  
    #
    # @param train_set_x        Training input x
    # @param train_set_y        Training output y
    # @param lossFunction       lossfunction to run gradient
    # @param learning_rate      update learning rate
    # @param n_epochs           maximum epoch to train
    # @param batch_size         how much data to train for one minibatch
    #
    # @return 
    def trainModel(self, train_set_x = None, train_set_y = None,
                   lossFunction = LossFunctions.LossFunctions.negative_log_likelihood,
                   learning_rate=0.13, n_epochs=1000,
                   batch_size=20):

        # if the model is empty, build the trainModel
        if (self._trainModel == None):
            if (train_set_x != None and train_set_y != None):
                self.buildTrainingModel(train_set_x, train_set_y, lossFunction,
                                        learning_rate,  batch_size)
            else:
                print "You have not build any training model yet."
                quit()

        # train the minibatchs
        for minibatch_index in xrange(self._totalBatches):
            minibatch_avg_cost = self._trainModel(minibatch_index)

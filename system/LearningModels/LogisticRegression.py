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

##
# @brief 
# Multi-class Logistic Regression Class
# 
# The logistic regression is fully described by a weight matrix :math:`W`
# and bias vector :math:`b`. Classification is done by projecting data
# points onto a set of hyperplanes, the distance to which is used to
# determine a class membership probability.
# 
class LogisticRegression(object):

    def __init__(self, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        self._x = T.matrix('x')  # data, presented as rasterized images
        self._y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        # TODO: change initialize value to configurable, eg. random
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
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
        self.p_y_given_x = T.nnet.softmax(T.dot(self._x, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model, will be used later to update
        # This can be stored and reload to replicate the same result
        self.params = [self.W, self.b]

        # initialize train model
        self._trainModel = None;

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    ##
    # @brief            Given a list of inputs , return a list of prediction
    #
    # @param testInput  
    #
    # @return 
    def testModel(self, testInput):
        # build test Model
        testInput = testInput.get_value()

        # loop through the input and compute prediction
        test_model = theano.function(
            inputs=[self._x],
            outputs=self.y_pred,
        )

        preditction = test_model(testInput)
        return preditction 


    ##
    # @brief                Given a list of inputs
    #
    # @param testInput      test input
    # @param testOutput     right output
    #
    # @return               error rate 
    def getTestError(self, testInput, testOutput):
        # compare the prediction with the output
        error = 0.0
        prediction = self.testModel(testInput)
        testOutput = testOutput.eval()
        for i in range(len(prediction)):
            if prediction[i] != testOutput[i]:
                error += 1

        return error / len(prediction)

    ##
    # @brief                Create theano function that take training x and y
    #                       and update W and b
    #
    # @param train_set_x    
    # @param train_set_y
    # @param lossFunction   The loss function that will calculate gradient. A function takes x and y
    # @param learning_rate  the step updateing parameters
    # @param batch_size     The size of training data to compute at once
    #
    # @return 
    def buildTrainingModel(self, train_set_x, train_set_y,
                           lossFunction = LossFunctions.LossFunctions.negative_log_likelihood, 
                           learning_rate = 0.1, batch_size = 600):

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        index = T.lscalar()  # index to a [mini]batch

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = lossFunction(self.p_y_given_x, self._y)

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)

        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(self.W, self.W - learning_rate * g_W),
                   (self.b, self.b - learning_rate * g_b)]

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

    def trainModel(self, train_set_x = None, train_set_y = None,
                   lossFunction = LossFunctions.LossFunctions.negative_log_likelihood,
                   learning_rate=0.13, n_epochs=1000,
                   batch_size=600):
        """
        Demonstrate stochastic gradient descent optimization of a log-linear
        model

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: the path of the MNIST dataset file from
                     http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

        """
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

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


def getRecurrentLayer(inputVariable, inputNumber, outputNumber, layerName = "rnn"):
    # parameters of the model, will be used later to update
    # This can be stored and reload to replicate the same result
    params = {}

    # initialize each layer and put them into params 

    # 1. From input layer to hiddent layer
    W_ih = theano.shared(
        value=numpy.asarray(
            ModelUtility.getRandomNumpyMatrix(inputNumber, outputNumber),
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    # initialize the baises b as a vector of n_out 0s
    b_ih = theano.shared(
        value=numpy.zeros(
            (outputNumber,),
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    # 2. From hidden layer to output layer
    W_ho = theano.shared(
        value=numpy.asarray(
            ModelUtility.getRandomNumpyMatrix(outputNumber, outputNumber),
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    # initialize the baises b as a vector of n_out 0s
    b_ho = theano.shared(
        value=numpy.zeros(
            (outputNumber,),
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    # initialize hidden layer to 0
    h0 = theano.shared(value=numpy.zeros(outputNumber,
                       dtype=theano.config.floatX))

    # project input variable to the same space as hidden layer
    inputVariable = T.tanh(T.dot(inputVariable, W_ih) + b_ih)
    # 3. construct recurrence
    def recurrence(x_t, h_tm1):
        h_t = x_t + h_tm1
        h_t = T.tanh(T.dot(h_t, W_ho) + b_ho)
        return h_t

    output, updates = theano.scan(fn=recurrence,
                                  sequences=inputVariable,
                                  outputs_info=[h0],
                                  n_steps=inputVariable.shape[0])

    params[layerName + "input_hidden_W"] = W_ih
    params[layerName + "input_hidden_b"] = b_ih
    params[layerName + "hidden_output_W"] = W_ho
    params[layerName + "hidden_output_b"] = b_ho

    return output, updates, params 
    

class RecurrentNN(Classifier.Classifier):

    # must have property of learning model
    # this will used to upload to database
    description = "h_t = s(x * W + b) + h_tm1, output = (h_t * W + b), class = logisticRegression(output)" 
    
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
    def __init__(self, n_in, n_out, layers = [5], name="rnn"):
        # initialize classifier class
        super(RecurrentNN, self).__init__()

        self.rnnOutput, self.rnnUpdates, self.rnnParams = getRecurrentLayer(self._x, n_in, layers[0], layerName = "rnn")         

        logisticRegressionInputNumber = layers[-1]
        self.p_y_given_x, logisticRegressionParams= LogisticRegression.getLogisticRegressionLayer(self.rnnOutput, logisticRegressionInputNumber, n_out) 

        self.params = {}
        for key in self.rnnParams:
            if key in self.params:
                assert("conflict key: " + key )
            self.params[key] = self.rnnParams[key]
        for key in logisticRegressionParams:
            if key in self.params:
                assert("conflict key: " + key )
            self.params[key] = logisticRegressionParams[key]

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


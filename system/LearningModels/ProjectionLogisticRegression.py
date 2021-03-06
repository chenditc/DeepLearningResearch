import os
import sys

import numpy

import theano
import theano.tensor as T

import LossFunctions
import ModelUtility
import Classifier
import ProjectionLayer
import LogisticRegression

class ProjectionLogisticRegression(Classifier.Classifier):

    # must have property of learning model
    # this will used to upload to database
    description = "y = projection[x] * W + b" 
    
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
        super(ProjectionLogisticRegression, self).__init__()

        # Remove hard coded maxIndex
        output, self.projParams = ProjectionLayer.getProjectionLayer(self._x, 253855, 50)

        self.p_y_given_x, self.params= LogisticRegression.getLogisticRegressionLayer(output, 50 * 5, 253855) 

        self.params.update(self.projParams)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


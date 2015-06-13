__docformat__ = 'restructedtext en'

import os
import sys
import math

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

import ModelUtility
import Classifier
import ProjectionLayer
import MultiWindowConvolutionLayer
import PoolingLayer
import ConvWordVectorLayer

 
class ConvWordVector(Classifier.Classifier):

    # must have property of learning model
    # this will used to upload to database
    description = "word_embdedding + Convolution + Pooling" 
    
    def __init__(self, maxWordCount, wordScanWindow, projectDimension):
        # initialize classifier class
        super(ConvWordVector, self).__init__()

        wordVector, wordVectorParams = ConvWordVectorLayer.getConvWordVectorLayer(self._x, classifier, wordScanWindow, projectDimension)

        self.p_y_given_x, logisticRegressionParams= LogisticRegression.getLogisticRegressionLayer(wordVector, projectDimension, maxWordCount) 

        self.params = {}
        self.params.update(wordVectorParams)
        self.params.update(logisticRegressionParams)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)



__docformat__ = 'restructedtext en'

import os
import sys

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

import ModelUtility
import Classifier
import ProjectionLayer
import MultiWindowConvolutionLayer

def getConvWordVectorLayer(inputVariable, maxWordCount, wordScanWindow = 5, wordVectorLength = 100 ):

    print "ConvWordVectorLayer:", " maxWordCount:", maxWordCount, " wordScanWindow:", wordScanWindow, "wordVectorLength", wordVectorLength

    outputs = None
    params = {}

    # 1. transform word index to word vector:
    wordVector, projectionParams = ProjectionLayer.getProjectionLayer(inputVariable, maxWordCount, wordVectorLength)
    params.update(projectionParams)

    # 1.1 add feature map dimension
    wordVector = wordVector.dimshuffle(0, 'x', 1, 2)

    # 2. add a multi-window-length convolution layer:
    convOut, convParams = MultiWindowConvolutionLayer.getMultiWindowConvolutionLayer(wordVector, wordScanWindow, wordVectorLength, wordVectorLength)
    params.update(convParams)

    outputs = convOut 
    return outputs, params



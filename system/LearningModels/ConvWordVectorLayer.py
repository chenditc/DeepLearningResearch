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

def getConvWordVectorLayer(inputVariable, maxWordCount, wordScanWindow = 5, wordVectorLength = 100, maximumWindowNumber = 2):

    print "ConvWordVectorLayer:", " maxWordCount:", maxWordCount, " wordScanWindow:", wordScanWindow, "wordVectorLength", wordVectorLength

    outputs = None
    params = {}

    # 1. transform word index to word vector:
    wordVector, projectionParams = ProjectionLayer.getProjectionLayer(inputVariable, maxWordCount, wordVectorLength)
    params.update(projectionParams)

    # 1.1 add feature map dimension
    wordVector = wordVector.dimshuffle(0, 'x', 1, 2)

    # 2. add a multi-window-length convolution layer:
    convOut, convParams = MultiWindowConvolutionLayer.getMultiWindowConvolutionLayer(wordVector, wordScanWindow, wordVectorLength, wordVectorLength, maximumWindowNumber = maximumWindowNumber)
    params.update(convParams)

    # 2.1 add a feature map dimension
    convOut = convOut.dimshuffle(0, 'x', 1, 2)

#    minibatch = inputVariable.shape[0]
#    poolingOut = convOut.reshape( (minibatch, sum(range(1, wordScanWindow + 1)) * wordVectorLength))

    # 3. add a pooling layer:
    poolingLength = sum(range(wordScanWindow - maximumWindowNumber + 1, wordScanWindow + 1)) 
    poolingOut, poolingParams = PoolingLayer.getPoolingLayer(convOut, poolingLength, mode='average_exc_pad')
    params.update(poolingParams)
   

    outputs = poolingOut 
    return outputs, params




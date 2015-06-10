__docformat__ = 'restructedtext en'

import os
import sys

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

import LossFunctions
import ModelUtility
import Classifier
import ConvolutionLayer


def getMultiWindowConvolutionLayer(inputVariable, windowHeight, windowWidth, featureMap, initFilterMatrixs = None):

    outputs = []
    params = {}
    for i in range(1, windowHeight+1): 
        tempOut, tempParams = ConvolutionLayer.getConvolutionLayer(inputVariable, i, windowWidth, featureMap, initFilterMatrixs[i-1], layerName = 'Conv-' + str(i))

        # condense each feature map to a vector, instead of matrix
        tempOut = tempOut.reshape((inputVariable.shape[0], featureMap, windowHeight + 1 - i))

        # for each window, get all feature to a vector
#        tempOut = tempOut.T

        outputs.append(tempOut)
        params.update(tempParams)
        
    convOut = T.concatenate(outputs, axis=2)
    convOut = convOut.dimshuffle((0,2,1))

    return convOut, params



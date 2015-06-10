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


def getConvolutionLayer(inputVariable, windowHeight, windowWidth, featureMap, initFilterMatrix = None, layerName = 'Conv'):
    rng = numpy.random.RandomState(23455)
    shape = (featureMap, 1, windowHeight, windowWidth)
    Filter = theano.shared( numpy.asarray(
                                rng.uniform( low=-0.1, high=0.1, size=shape),
                                dtype=theano.config.floatX), 
                            name = layerName + '-Filter',
                            borrow=True)

    if initFilterMatrix != None:
        Filter.set_value(initFilterMatrix)

    convOut = conv.conv2d(inputVariable, Filter)

    params = {
        layerName + '-Filter' : Filter, 
    }

    return convOut, params

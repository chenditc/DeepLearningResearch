__docformat__ = 'restructedtext en'

import os
import sys

import numpy

import theano
from theano.tensor.signal import downsample

import LossFunctions
import ModelUtility
import Classifier


def getPoolingLayer(inputVariable, windowHeight, mode='average_exc_pad', layerName = 'Pooling'):
    shape = (windowHeight, 1)

    poolingOut = downsample.max_pool_2d(inputVariable, shape, ignore_border=False, mode=mode)
    poolingOut = poolingOut.flatten(ndim=2)

    params = {
    }

    return poolingOut, params


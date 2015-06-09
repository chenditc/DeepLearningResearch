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
#
# @param inputVariable      The indexes of inputs, should be an array
# @param maxIndex           maximum index number
# @param projectDimension   target dimension projecting
#
# @return 
def getProjectionLayer(inputVariable, maxIndex, projectDimension, initProjectionMatrix = None):
    Projection = theano.shared(
        value=numpy.asarray(
            ModelUtility.getRandomNumpyMatrix(maxIndex + 1, projectDimension), # add one for padding
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    if initProjectionMatrix != None:
        Projection.set_value(initProjectionMatrix)

    inputVariable = T.cast(inputVariable, 'int32')

    vectors = Projection[inputVariable]
    newShape = (inputVariable.shape[0], inputVariable.shape[1] * Projection.shape[1])
    reshapredVector = vectors.reshape(newShape) 

    params = {
        'Projection' : Projection, 
    }

    return reshapredVector, params

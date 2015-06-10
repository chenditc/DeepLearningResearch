#!/usr/bin/python

import numpy
import theano
import theano.tensor as T
import ConvolutionLayer 

initFilterMatrix = numpy.asarray(
       [ [[[0,0,0,0,0],
           [0,0,0,0,0]]],

         [[[1,1,1,1,1],
           [1,1,1,1,1]]],

         [[[2,2,2,2,2],
           [2,2,2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)

inputVariable = T.tensor4(name='input') 

windowHeight = 2
windowWidth = 5
featureMap = 3

convOut, params = ConvolutionLayer.getConvolutionLayer(inputVariable, windowHeight, windowWidth, featureMap, initFilterMatrix)

f = theano.function(
                inputs=[inputVariable],
                outputs=convOut,
                allow_input_downcast=True
)

inputArray = numpy.asarray(
        [[[[1,2,3,4,5],
              [2,3,4,5,6],
              [3,4,5,6,7],
              [4,5,6,7,8],
              [5,6,7,8,9]]]]
        ) 
output = f(inputArray)

# condense each feature map to a vector, instead of matrix
output = output.reshape((featureMap,windowHeight-1))

# for each window, get all feature to a vector
output = output.T

print output
print params['Filter'].get_value()

standard = [[   0,   35,   70],
            [   0,   45,   90],
            [   0,   55,  110],
            [   0,   65,  130]]

assert(numpy.array_equal(output,standard))


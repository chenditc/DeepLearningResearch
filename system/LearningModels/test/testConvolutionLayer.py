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
output = output.reshape((1, featureMap, 5 + 1 - windowHeight))



print output
print params['Conv-Filter'].get_value()

standard = [[[   0,    0,    0,    0],
             [  35,   45,   55,   65],
             [  70,   90,  110,  130]]]

assert(numpy.array_equal(output,standard))


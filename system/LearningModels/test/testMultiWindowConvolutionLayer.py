#!/usr/bin/python

import numpy
import theano
import theano.tensor as T
import MultiWindowConvolutionLayer

initFilterMatrix0 = numpy.asarray(
       [ [[[1,1,1,1,1]]],

         [[[2,2,2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)
initFilterMatrix1 = numpy.asarray(
       [ [[[1,1,1,1,1],
           [1,1,1,1,1]]],

         [[[2,2,2,2,2],
           [2,2,2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)

initFilterMatrix2 = numpy.asarray(
       [ [[[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1]]],

         [[[2,2,2,2,2],
           [2,2,2,2,2],
           [2,2,2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)

initFilterMatrix3 = numpy.asarray(
       [ [[[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1]]],

         [[[2,2,2,2,2],
           [2,2,2,2,2],
           [2,2,2,2,2],
           [2,2,2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)

initFilterMatrix4 = numpy.asarray(
       [ [[[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1]]],

         [[[2,2,2,2,2],
           [2,2,2,2,2],
           [2,2,2,2,2],
           [2,2,2,2,2],
           [2,2,2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)

inputVariable = T.tensor4(name='input') 

windowHeight = 5
windowWidth = 5
featureMap = 2

convOut, params = MultiWindowConvolutionLayer.getMultiWindowConvolutionLayer(inputVariable, windowHeight, windowWidth, featureMap, [initFilterMatrix0, initFilterMatrix1, initFilterMatrix2, initFilterMatrix3, initFilterMatrix4])

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

print output
print params

standard = [[  15,   30],
            [  20,   40],
            [  25,   50],
            [  30,   60],
            [  35,   70],
            [  35,   70],
            [  45,   90],
            [  55,  110],
            [  65,  130],
            [  60,  120],
            [  75,  150],
            [  90,  180],
            [  90,  180],
            [ 110,  220],
            [ 125,  250]]

assert(numpy.array_equal(output,standard))
assert(len(params) == 5)

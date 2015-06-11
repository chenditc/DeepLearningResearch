#!/usr/bin/python

import numpy
import theano
import theano.tensor as T
import PoolingLayer 

inputVariable = T.tensor4(name='input') 

poolingOut, params = PoolingLayer.getPoolingLayer(inputVariable, 15, mode='max')

f = theano.function(
                inputs=[inputVariable],
                outputs=poolingOut,
                allow_input_downcast=True
)

inputArray = numpy.asarray(
          [[[[  15,   30],
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
             [ 125,  250]]]]
        ) 
output = f(inputArray)

print output
print params

standard = numpy.reshape(numpy.max(inputArray, axis=2), (2))

assert(numpy.array_equal(output,standard))
assert(len(params) == 0)

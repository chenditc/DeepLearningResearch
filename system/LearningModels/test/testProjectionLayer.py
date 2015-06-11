#!/usr/bin/python

import numpy
import theano
import theano.tensor as T
import ProjectionLayer 

Projection = numpy.asarray(
        [[1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12],
         [13,14,15]], # add one for padding
        dtype=theano.config.floatX
)

inputVariable = T.matrix() 
maxIndex = 5    
projectDimension = 3    # how long is each vector

reshapredVector, params = ProjectionLayer.getProjectionLayer(inputVariable, maxIndex, projectDimension, Projection)

f = theano.function(
                inputs=[inputVariable],
                outputs=reshapredVector,
)

output = f([[1,3],
         [2,4]])

print output
standard = [
            [[4,5,6],
             [10,11,12]],

            [[7,8,9],
             [13,14,15]]]

assert(numpy.array_equal(output,standard))


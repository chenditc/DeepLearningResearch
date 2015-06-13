#!/usr/bin/python

import numpy
import theano
import theano.tensor as T
import ConvWordVectorLayer

Projection = numpy.asarray(
        [[1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12],
         [13,14,15]], # add one for padding
        dtype=theano.config.floatX
)

initFilterMatrix0 = numpy.asarray(
       [ [[[1,1,1]]],
         [[[3,3,3]]],
         [[[2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)
initFilterMatrix1 = numpy.asarray(
       [ [[[1,1,1],
           [1,1,1]]],
         [[[3,3,3],
           [3,3,3]]],
         [[[2,2,2],
           [2,2,2]]] ], # add one for padding
        dtype=theano.config.floatX
)

inputVariable = T.matrix() 
maxIndex = 5    
projectDimension = 3    # how long is each vector
wordScanWindow = 2

reshapredVector, params = ConvWordVectorLayer.getConvWordVectorLayer(inputVariable,  maxIndex, wordScanWindow, projectDimension)

params['Projection'].set_value(Projection)
params['Conv-1-Filter'].set_value(initFilterMatrix0)
params['Conv-2-Filter'].set_value(initFilterMatrix1)

f = theano.function(
                inputs=[inputVariable],
                outputs=reshapredVector,
                mode='DebugMode'
)

output = f([[1,3],
         [2,4]])

print output
print params
standard = [[  32,   96,   64],
            [  44,  132,   88]]

assert(numpy.array_equal(output,standard))


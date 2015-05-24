#!/usr/bin/python -u

# This module include the process of training a deep learning model
import argparse
import sys
import json
import pprint

import numpy
import theano
import theano.tensor as T
import storm

import LogisticRegression
import MultilayerPerceptron
import RecurrentNN
import LossFunctions
import DataLoader


class TestBolt(storm.BasicBolt):
    def initialize(self, stormconf, context):
        model_id = "LogisticRegression" 
        self.tester = TestModel(data_id = "abc", model_id = model_id)

    def getXYFromTuple(self, tup):
        x = tup.values[0]
        y = tup.values[1]
        x = numpy.asarray(json.loads(x))
        y = numpy.asarray(json.loads(y))
        return x, y


    def process(self, tup):
        try:
            x, y = self.getXYFromTuple(tup)
            result = self.tester.startTesting(x, y)
            storm.log("Testing Result:" + str(result))        
        except:
            storm.log("Unexpected error:" + str(sys.exc_info()[0]))

class TestModel:

    def __init__(self, data_id, model_id):
        # Get required meta data from data set, eg. dimensionality
#        inputDim, outputDim = dataLoader.getDataDimension() 
        inputDim = 2
        outputDim = 2

        classifierModule = __import__(model_id)
        classifierClass = getattr(classifierModule, model_id)
        self.model = classifierClass(n_in = inputDim, n_out = outputDim)
        
    def startTesting(self, x, y):
        testResult = self.model.getTestError(x, y)
        return testResult 


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Testing Entrance.')
    parser.add_argument('-d', '--data_id', dest='data_id', help='the data used to train')
    parser.add_argument('-m', '--model_id', dest='model_id', help='the model used to train')


    args = parser.parse_args()

    if (args.data_id == None or args.model_id == None ):
        TestBolt().run()
        quit()


    tester = TestModel(data_id = args.data_id, model_id = args.model_id)
    result = tester.startTesting([[-100,-100]], [0])
    storm.log("Testing Result:" + str(result))        

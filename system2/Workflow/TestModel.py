#!/usr/bin/python -u

# This module include the process of training a deep learning model
import argparse
import sys
import json
import pprint
import Queue

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
    def __init__(self, data_id, model_id):
        self.data_id = data_id
        self.model_id = model_id

        # Set default
        if self.model_id == None:
            self.model_id = "LogisticRegression" 
        if self.data_id == None:
            self.data_id = "sum_positive_1"

    def initialize(self, stormconf, context):
        self.tester = TestModel(data_id = self.data_id, model_id = self.model_id)
        self.testResult = []

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

            self.testResult.append(result)
            if len(self.testResult) > 30:
                self.testResult.pop(0)

            storm.log("Testing Result:" + str( sum(self.testResult)/len(self.testResult) ))        

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
    parser.add_argument('-b', '--bolt', dest='bolt', action="store_true", help='create a storm bolt')

    args = parser.parse_args()

    if (args.data_id == None or args.model_id == None ):
        parser.print_help()
        quit()

    if (args.bolt):
        testBolt = TestBolt(args.data_id, args.model_id)
        testBolt.run()


    tester = TestModel(data_id = args.data_id, model_id = args.model_id)
    result = tester.startTesting([[-100,-100]], [0])
    storm.log("Testing Result:" + str(result))        

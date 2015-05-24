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

import Model
import LogisticRegression
import MultilayerPerceptron
import RecurrentNN
import LossFunctions
import TestModel
import DataLoader


class GradientBolt(storm.BasicBolt):
    def initialize(self, stormconf, context):
        model_id = "LogisticRegression" 
        self.trainer = TrainModel(data_id = "abc", model_id = model_id)

    def getXYFromTuple(self, tup):
        x = tup.values[0]
        y = tup.values[1]
        x = numpy.asarray(json.loads(x))
        y = numpy.asarray(json.loads(y))
        storm.log("x is:")
        storm.log(str(x))
        return x, y


    def process(self, tup):
        try:
            x, y = self.getXYFromTuple(tup)
            gradientsName, gradients = self.trainer.startTraining(x, y)

            for i in range(len(gradientsName)):
                storm.emit([json.dumps(gradientsName[i]), json.dumps(gradients[i].tolist())])
        except:
            storm.log("Unexpected error:" + str(sys.exc_info()[0]))


class TrainModel:

    def __init__(self, data_id, model_id):
        # Get required meta data from data set, eg. dimensionality
#        inputDim, outputDim = dataLoader.getDataDimension() 
        inputDim = 2
        outputDim = 2

        self.model = Model.Model.loadModelByName(model_id, inputDim, outputDim)

        
    def startTraining(self, x, y):
        gradientsName, gradients = self.model.trainModel(x, y)
        return gradientsName, gradients 


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Training Entrance.')
    parser.add_argument('-d', '--data_id', dest='data_id', help='the data used to train')
    parser.add_argument('-m', '--model_id', dest='model_id', help='the model used to train')


    args = parser.parse_args()

    if (args.data_id == None or args.model_id == None ):
        GradientBolt().run()
        quit()


    trainer = TrainModel(data_id = args.data_id, model_id = args.model_id)
    gradientsName, gradients = trainer.startTraining([[2,2]],[0])
    print json.dumps(gradientsName)
    print gradients
    gradients = [numpy.asarray(x).tolist() for x in gradients]
    print json.dumps(gradients)
    


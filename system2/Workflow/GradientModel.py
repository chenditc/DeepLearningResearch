#!/usr/bin/python -u

# This module include the process of training a deep learning model
import traceback
import argparse
import sys
import json
import pprint

import MySQLdb
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
    def __init__(self, data_id, model_id):
        self.data_id = data_id
        self.model_id = model_id

        # Set default
        if self.model_id == None:
            self.model_id = "LogisticRegression" 
        if self.data_id == None:
            self.data_id = "sum_positive_1"

    def initialize(self, stormconf, context):
        try:
            self.trainer = TrainModel(data_id = self.data_id, model_id = self.model_id)
        except:
            storm.log("Failed to initialize model")
            storm.log("Unexpected error:" + str(sys.exc_info()[0]))
            storm.log(traceback.format_exc())


    def getXYFromTuple(self, tup):
        x = tup.values[0]
        y = tup.values[1]
        x = numpy.asarray(json.loads(x))
        y = numpy.asarray(json.loads(y))
        return x, y


    def process(self, tup):
        try:
            x, y = self.getXYFromTuple(tup)
            gradientsName, gradients = self.trainer.startTraining(x, y)

            for i in range(len(gradientsName)):
                storm.emit([json.dumps(gradientsName[i]), json.dumps(gradients[i].tolist())])
        except:
            storm.log("Unexpected error:" + str(sys.exc_info()[0]))
            storm.log(traceback.format_exc())


class TrainModel:

    def __init__(self, data_id, model_id):
        # Get required meta data from data set, eg. dimensionality
        dbConnector = MySQLdb.connect(host="deeplearningdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                      user="research", 
                                      passwd="Research013001",
                                      db="DeepLearningDB1")
        dbCursor = dbConnector.cursor() 
        dbCursor.execute('SELECT inputDimension, outputDimension FROM TrainingDataMetaData1 WHERE data_id = %s', (data_id) )
        dataRows = dbCursor.fetchall()
        inputDim = dataRows[0][0]
        outputDim = dataRows[0][1]

        self.model = Model.Model.loadModelByName(model_id, inputDim, outputDim, taskName = data_id + "-" + model_id)

        
    def startTraining(self, x, y):
        gradientsName, gradients = self.model.trainModel(x, y)
        return gradientsName, gradients 


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Training Entrance.')
    parser.add_argument('-d', '--data_id', dest='data_id', help='the data used to train')
    parser.add_argument('-m', '--model_id', dest='model_id', help='the model used to train')
    parser.add_argument('-b', '--bolt', dest='bolt', action="store_true", help='create a storm bolt')

    args = parser.parse_args()

    if (args.data_id == None or args.model_id == None ):
        parser.print_help()
        quit()

    if (args.bolt):
        gradientBolt = GradientBolt(args.data_id, args.model_id)
        gradientBolt.run()

    trainer = TrainModel(data_id = args.data_id, model_id = args.model_id)
    gradientsName, gradients = trainer.startTraining([[2,2]],[0])
    print json.dumps(gradientsName)
    print gradients
    gradients = [numpy.asarray(x).tolist() for x in gradients]
    print json.dumps(gradients)
    


#!/usr/bin/python

# This module include the process of training a deep learning model
import argparse
import sys
import json
import pprint

import numpy
import theano
import theano.tensor as T

import LogisticRegression
import MultilayerPerceptron
import RecurrentNN
import LossFunctions
import EarlyStopTrainer
import TestModel
import DataLoader



class TrainModel:

    def __init__(self, data_id, model_id, config):
        # private variables:
        self._data_id = data_id
        self._model_id = model_id
        self._config = config


    def startTraining(self, isClassifier = True):
        # Use Dataloader class to load data set.
        print "#####################################"
        print "Loading data: ", self._data_id
        dataLoader = DataLoader.DataLoader(dataset = self._data_id, config = self._config)

        print "#####################################\n"
        # Get required meta data from data set, eg. dimensionality
        inputDim, outputDim = dataLoader.getDataDimension() 
        print "Dimensions:" , inputDim, outputDim

        # Create training model
        print "#####################################"

        print "Initializing model: ", self._model_id
        classifierModule = __import__(self._model_id)
        classifierClass = getattr(classifierModule, self._model_id)
        classifier = classifierClass(n_in = inputDim, n_out = outputDim)


        
        # Initialize trainer, here we use Early stopping 
        trainer = EarlyStopTrainer.EarlyStopTrainer(classifier, dataLoader, config = self._config) 

        # Use trainer to train model
        trainer.trainModel()

        # Call testModel(testSet) to test the model
        tester = TestModel.TestModel(model = classifier)
        tester.testModel()

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Training Entrance.')
    parser.add_argument('-d', '--data_id', dest='data_id', help='the data used to train')
    parser.add_argument('-m', '--model_id', dest='model_id', help='the model used to train')
    parser.add_argument('-c', '--config', dest='configFile', default = './config', help='config file defines few training parameters')


    args = parser.parse_args()

    if (args.data_id == None or args.model_id == None ):
        parser.print_help()
        quit()


    # open onfig file and load to a map
    configFile = open(args.configFile).read()
    config = json.loads(configFile)
    pprinter = pprint.PrettyPrinter(indent=4)
    pprinter.pprint(config)

    trainer = TrainModel(data_id = args.data_id, model_id = args.model_id, config = config)
    trainer.startTraining()

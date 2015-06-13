#!/usr/bin/python -u

# This module include the process of training a deep learning model
import argparse
import sys
import json
import pprint
import codecs

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


class TextLoader():
    def __init__(self, dataDir, indexToWord):
        self._dataDir = dataDir

        # TODO: load indexToWord and then contruct wordToIndex
        

    def __iter__(self):
        nextData = []
        for filename in os.listdir(self._dataDir):
            for line in codecs.open(os.path.join(self._dataDir, filename), encoding='utf-8'):
                oneSentence = list(line)

                # TODO: Use wordToIndex Contruct matrixes

                yield nextData
                nextData = []

class TrainModel():

    def __init__(self, data):
        # private variables:
        self._data = data


    def startTraining(self):
        # Use Dataloader class to load data set.
        print "#####################################"
        print "Loading data: ", self._data
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
    parser.add_argument('-d', '--data', dest='data', help='the data used to train')


    args = parser.parse_args()

    if (args.data == None):
        parser.print_help()
        quit()


    trainer = TrainModel(data = args.data)
    trainer.startTraining()

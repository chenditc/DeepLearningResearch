#!/usr/bin/python

# This module include the process of training a deep learning model

import numpy
import theano
import theano.tensor as T

import LogisticRegression
import MultilayerPerceptron
import LossFunctions
import EarlyStopTrainer
import TestModel
import DataLoader



class TrainModel:

    def __init__(self, data_id):
        # private variables:
        self._data_id = data_id


    def startTraining(self, isClassifier = True):
        # Use Dataloader class to load data set.
        print "#####################################"
        print "Loading data: ", self._data_id
        dataLoader = DataLoader.DataLoader(dataset=self._data_id)

        print "#####################################\n"
        # Get required meta data from data set, eg. dimensionality
        inputDim, outputDim = dataLoader.getDataDimension() 
        print inputDim, outputDim

        # Create training model
        print "#####################################"
#        print "Initializing model: ", LogisticRegression.LogisticRegression.__name__
#        classifier = LogisticRegression.LogisticRegression(n_in = inputDim, n_out = outputDim)
        print "Initializing model: ", MultilayerPerceptron.MultilayerPerceptron.__name__
        classifier = MultilayerPerceptron.MultilayerPerceptron(n_in = inputDim, n_out = outputDim)

        
        # Initialize trainer, here we use Early stopping 
        trainer = EarlyStopTrainer.EarlyStopTrainer(classifier, dataLoader) 

        # Use trainer to train model
        trainer.trainModel()

        # Call testModel(testSet) to test the model
        tester = TestModel.TestModel(model = classifier)
        tester.testModel()

if __name__ == "__main__" :
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Training Entrance.')
    parser.add_argument('--data_id', dest='data_id', help='the data used to train')
    args = parser.parse_args()

    if (args.data_id == None):
        parser.print_help()
        quit()


    trainer = TrainModel(data_id = args.data_id)
    trainer.startTraining()

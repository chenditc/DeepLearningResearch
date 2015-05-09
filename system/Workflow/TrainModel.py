#!/usr/bin/python

# This module include the process of training a deep learning model

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

    def __init__(self, data_id, model_id):
        # private variables:
        self._data_id = data_id
        self.model_id = model_id


    def startTraining(self, isClassifier = True):
        # Use Dataloader class to load data set.
        print "#####################################"
        print "Loading data: ", self._data_id
        dataLoader = DataLoader.DataLoader(dataset=self._data_id)

        print "#####################################\n"
        # Get required meta data from data set, eg. dimensionality
        inputDim, outputDim = dataLoader.getDataDimension() 
        print "Dimensions:" , inputDim, outputDim

        # Create training model
        print "#####################################"
#        print "Initializing model: ", LogisticRegression.LogisticRegression.__name__
#        classifier = LogisticRegression.LogisticRegression(n_in = inputDim, n_out = outputDim)
#        print "Initializing model: ", MultilayerPerceptron.MultilayerPerceptron.__name__
#        classifier = MultilayerPerceptron.MultilayerPerceptron(n_in = inputDim, n_out = outputDim)
#        print "Initializing model: ", RecurrentNN.RecurrentNN.__name__
#        classifier = RecurrentNN.RecurrentNN(n_in = inputDim, n_out = outputDim)

        print "Initializing model: ", self.model_id
        classifierModule = __import__(self.model_id)
        classifierClass = getattr(classifierModule, self.model_id)
        classifier = classifierClass(n_in = inputDim, n_out = outputDim)


        
        # Initialize trainer, here we use Early stopping 
        trainer = EarlyStopTrainer.EarlyStopTrainer(classifier, dataLoader, batch_size = 50) 

        # Use trainer to train model
        trainer.trainModel()

        # Call testModel(testSet) to test the model
        tester = TestModel.TestModel(model = classifier)
        tester.testModel()

if __name__ == "__main__" :
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Training Entrance.')
    parser.add_argument('-d', '--data_id', dest='data_id', help='the data used to train')
    parser.add_argument('-m', '--model_id', dest='model_id', help='the model used to train')

    args = parser.parse_args()

    if (args.data_id == None or args.model_id == None ):
        parser.print_help()
        quit()


    trainer = TrainModel(data_id = args.data_id, model_id = args.model_id)
    trainer.startTraining()

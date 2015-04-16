#!/usr/bin/python

# This module include the process of training a deep learning model

import numpy
import theano
import theano.tensor as T

import LogisticRegression
import LossFunctions
import EarlyStopTrainer
import TestModel

class TrainModel:

    def __init__(self, data_id):
        # private variables:
        self._data_id = data_id


    def load_data(self, dataset):
        ''' Loads the dataset

        :type dataset: string
        :param dataset: the path to the dataset (here MNIST)
        '''

        #############
        # LOAD DATA #
        #############

        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.
        import DataLoader
        dataLoader = DataLoader.DataLoader(dataset=dataset)

        train_set_x, train_set_y = dataLoader.getTrainingSet() 
        valid_set_x, valid_set_y = dataLoader.getValidationSet() 
        test_set_x, test_set_y = dataLoader.getTestSet() 

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval



    def startTraining(self, isClassifier = True):
        # Use Dataloader class to load data set.
        print "#####################################"
        print "Loading data: ", self._data_id
        datasets = self.load_data(self._data_id)
        print "#####################################"

        print "#####################################"
        print "Calcualting model setting" 
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        xValue = train_set_x.get_value()
        yValue = train_set_y.eval()

        # Get required meta data from data set, eg. dimensionality
        inputDim = len(xValue[0])
        outputDim = 0
        if (isClassifier):
            outputDim = max(yValue) + 1
        else:
            outputDim = len(yValue[0])
        xValue = None
        yValue = None
        print "#####################################"



        # Create training model
        print "#####################################"
        print "Initializing model: ", LogisticRegression.LogisticRegression.__name__
        classifier = LogisticRegression.LogisticRegression(n_in = inputDim, n_out = outputDim)
        
        # Initialize trainer, here we use Early stopping 
        trainer = EarlyStopTrainer.EarlyStopTrainer(classifier, train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y) 

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

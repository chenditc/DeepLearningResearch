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



    ##
    # @brief 
    #    Function that loads the dataset into shared variables
    #
    #    The reason we store our dataset in shared variables is to allow
    #    Theano to copy it into the GPU memory (when code is run on GPU).
    #    Since copying data into the GPU is slow, copying a minibatch everytime
    #    is needed (the default behaviour if the data is not in a shared
    #    variable) would lead to a large decrease in performance.
    #    
    # @param inputData      the input data to model 
    # @param outputData     the output data to model
    # @param borrow         if enable shallow copy or not
    # @param isClassifier   if this is a classifier
    #
    # @return 
    @staticmethod
    def shared_dataset(inputData, outputData, borrow=True, isClassifier = True):
        data_x = inputData
        data_y = outputData

        # if the data is for a classifier, use the first column of y only
        # and also change it to an array of scalar
        if isClassifier:
            data_y = data_y[:,0]

        # Create share variable from numpy array
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

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
        dataLoader = DataLoader.DataLoader()
        dataLoader.downloadData(dataset)
        trainingInput, trainingOutput = dataLoader.getTrainingSet() 
        validationInput, validationOutput = dataLoader.getValidationSet() 
        testInput, testOutput = dataLoader.getTestSet() 


        # Create 
        train_set_x, train_set_y = TrainModel.shared_dataset(trainingInput, trainingOutput)
        valid_set_x, valid_set_y = TrainModel.shared_dataset(validationInput, validationOutput)
        test_set_x, test_set_y = TrainModel.shared_dataset(testInput, testOutput)

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

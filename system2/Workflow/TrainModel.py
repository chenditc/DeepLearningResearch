#!/usr/bin/python -u

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
import TestModel
import DataLoader



class TrainModel:

    def __init__(self, data_id, model_id, config):
        # Get required meta data from data set, eg. dimensionality
#        inputDim, outputDim = dataLoader.getDataDimension() 
        inputDim = 2
        outputDim = 2

        classifierModule = __import__(model_id)
        classifierClass = getattr(classifierModule, model_id)
        self.model = classifierClass(n_in = inputDim, n_out = outputDim)
        
    def startTraining(self, x, y):
        gradientsName, gradients = self.model.trainModel(x, y)
        print gradientsName, gradients 


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

    trainer = TrainModel(data_id = args.data_id, model_id = args.model_id, config = config)
    trainer.startTraining([[2,2]],[0])

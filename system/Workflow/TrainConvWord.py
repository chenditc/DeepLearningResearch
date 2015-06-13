#!/usr/bin/python -u
#-*- coding: utf-8 -*-

# This module include the process of training a deep learning model
import argparse
import sys
import json
import os
import logging
import pprint
import codecs
import cPickle
import signal
import time

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
import ConvWordVector


class TextLoader():
    def __init__(self, dataDir, warmupModel, windowSize = 5, batchSize = 10000):
        self._dataDir = dataDir

        warmupModelString = open(warmupModel).read()
        indexToWordArray = cPickle.loads(warmupModelString)['index2word']
        self._wordToIndex = dict([(indexToWordArray[i], i) for i in range(len(indexToWordArray))])

        self._windowSize = windowSize
        self._batchSize = batchSize

    def __iter__(self):
        nextX = []
        nextY = []
        for filename in os.listdir(self._dataDir):
            for line in codecs.open(os.path.join(self._dataDir, filename), encoding='utf-8'):
                oneSentence = list(line)
                for yIndex in range(self._windowSize + 1, len(oneSentence)):
                    try:
                        tempX = [self._wordToIndex[oneSentence[xIndex]] for xIndex in range(yIndex - self._windowSize, yIndex)]
                        tempY = self._wordToIndex[oneSentence[yIndex]]
                        nextX.append(tempX)
                        nextY.append(tempY)
                        if len(nextX) >= self._batchSize:
                            yield nextX, nextY
                            nextX = []
                            nextY = []
                    except KeyError:
                        logging.info("Unknown word")
                        continue

class TrainModel():

    def __init__(self, data, warmupModel):
        # private variables:
        # Use Dataloader class to load data set.
        logging.info("#####################################")
        logging.info("Loading data: " + data)
        self._textLoader = TextLoader(data, warmupModel)
        self._windowSize = self._textLoader._windowSize

        # load pre-trained word vector
        warmupModelString = open(warmupModel).read()
        self._wordMatrix = cPickle.loads(warmupModelString)['wordMatrix']

        # handle sigterm
        def sigterm_handler(_signo, _stack_frame):
            # Raises SystemExit(0):
            sys.exit(0)
        signal.signal(signal.SIGTERM, sigterm_handler)



    def startTraining(self):
        # Create training model
        logging.info("#####################################")
        logging.info("Initializing model ")
        classifier = ConvWordVector.ConvWordVector(maxWordCount = len(self._wordMatrix) , wordScanWindow = self._windowSize, projectDimension = len(self._wordMatrix[0]))

        
        # build model
        train_set_x = None;
        train_set_y = None;
        learning_rate = None;

        try:
            t2 = 0
            t1 = 0
            for newX, newY in self._textLoader:
                if train_set_x == None:
                    train_set_x = theano.shared(
                        value=numpy.asarray(
                            newX,
                            dtype=theano.config.floatX
                        ),
                        borrow=True
                    )
                    train_set_y = theano.shared(
                        value=numpy.asarray(
                            newY,
                            dtype='int32'
                        ),
                        borrow=True
                    )
                    learning_rate = theano.shared(value=0.05, borrow=True)
                    classifier.buildTrainingModel(train_set_x, train_set_y, batch_size=400)
                    classifier.params['Projection'].set_value(numpy.asarray(self._wordMatrix), borrow=True)
                    logging.info("Build model finished")
                else:
                    train_set_x.set_value(newX)
                    train_set_y.set_value(newY)
                
                t1 = time.time() 
                logging.info("Preprocessing time {0}".format(t1-t2))
                classifier.trainModel()
                t2 = time.time()
                if learning_rate.get_value(borrow=True) > 0.005:
                    learning_rate.set_value(learning_rate.get_value(borrow=True) * 0.99, borrow=True)
                logging.info("Trained used time {0}, learning rate: {1}".format(t2-t1, learning_rate.get_value(borrow=True)))
        finally:
            modelName = '/home/ubuntu/data/convWordVector.model'
            modelString = classifier.storeModelToJson()
            outputFile = open(modelName, 'w')
            outputFile.write(modelString)
            outputFile.close()


if __name__ == "__main__" :

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Training Entrance.')
    parser.add_argument('-d', '--data', dest='data', help='the data used to train')
    parser.add_argument('-i', '--indexToWord', dest='indexToWord', help='the indexToWord cPickle file')

    args = parser.parse_args()

    if (args.data == None or args.indexToWord == None):
        parser.print_help()
        quit()


    trainer = TrainModel(data = args.data, warmupModel = args.indexToWord)
    trainer.startTraining()


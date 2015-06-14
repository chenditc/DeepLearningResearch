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
import scipy
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
import ConvolutionLayer

class TestModel():

    def __init__(self, data, warmupModel):
        # load word index mapping
        warmupModelString = open(warmupModel).read()
        self.indexToWord = cPickle.loads(warmupModelString)['index2word']
        self._wordToIndex = dict([(self.indexToWord[i], i) for i in range(len(self.indexToWord))])

        # load pre-trained word vector
        parameterMap = json.loads(open(data).read())
        self.wordVector = parameterMap['Projection']
        self.one_gram = parameterMap['Conv-1-Filter']

    def startTesting(self):
        inputVariable = T.tensor4(name='input') 
        tempOut, tempParams = ConvolutionLayer.getConvolutionLayer(inputVariable, 1, 100, 100, self.one_gram, layerName = 'Conv')
        tempOut = tempOut.flatten()
        f = theano.function([inputVariable], tempOut)

        # contructing word map
        for word in range(len(self.wordVector)):
            self.wordVector[word] = f([[[self.wordVector[word]]]])

        print "#### start testing ###"

        while True:
            line = raw_input().decode(sys.stdin.encoding) 
            words = list(line)
            word1 = words[0]
            word2 = words[1]
            print "----Testing {0} ------".format(words)
            wordVector1 = self.wordVector[self._wordToIndex[word1]]
            wordVector2 = self.wordVector[self._wordToIndex[word2]]

            print numpy.dot(wordVector1, wordVector2) / (numpy.sqrt(numpy.dot(wordVector1, wordVector1)) * numpy.sqrt(numpy.dot(wordVector2, wordVector2)) ) 

#            wordVector1 = numpy.cross(self.one_gram,wordVector1) 
#            wordVector2 = numpy.cross(self.one_gram,wordVector2) 
#            print numpy.dot(wordVector1, wordVector2)




if __name__ == "__main__" :

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Test word mapping')
    parser.add_argument('-d', '--data', dest='data', help='the word vector data')
    parser.add_argument('-i', '--indexToWord', dest='indexToWord', help='the indexToWord cPickle file')

    args = parser.parse_args()

    if (args.data == None or args.indexToWord == None):
        parser.print_help()
        quit()


    tester = TestModel(data = args.data, warmupModel = args.indexToWord)
    tester.startTesting()


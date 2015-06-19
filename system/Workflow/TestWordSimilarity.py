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
import heapq

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
import ConvWordVectorLayer


def getSim(wordVector1, wordVector2):
    return numpy.dot(wordVector1, wordVector2) / (numpy.sqrt(numpy.dot(wordVector1, wordVector1)) * numpy.sqrt(numpy.dot(wordVector2, wordVector2)) ) 

class TestModel():

    def __init__(self, data, warmupModel):
        # load word index mapping
        warmupModelString = open(warmupModel).read()
        self.indexToWord = cPickle.loads(warmupModelString)['index2word']
        self._wordToIndex = dict([(self.indexToWord[i], i) for i in range(len(self.indexToWord))])

        # load pre-trained word vector
        self.parameterMap = json.loads(open(data).read())
        self.wordVector = numpy.asarray(self.parameterMap['Projection'])

    def getWordVectors(self, words):
        if len(words) > 5:
            print "words too long"
            return []
        indexes = [self._wordToIndex[word] for word in words]
        print "----Testing {0} ------".format(words)
        wordVector = self.vectorFunction[len(words)-1]([indexes]) 
        wordVector = wordVector[0]
        return wordVector

    def startTesting(self):
        self.vectorFunction = []
        # TODO: change hard code filter number
        for i in range(1, 3):
            inputVariable = T.matrix() 
            tempOut, tempParams = ConvWordVectorLayer.getConvWordVectorLayer(inputVariable, len(self.wordVector), i, len(self.wordVector[0]))
#            inputVariable = T.tensor4(name='input')
#            tempOut, tempParams = ConvolutionLayer.getConvolutionLayer(inputVariable, i, len(self.wordVector[0]), len(self.wordVector[0]), layerName = 'Conv-'+str(i))
#            tempOut = tempOut.reshape((inputVariable.shape[0], len(self.wordVector[0])))
            tempOut = tempOut.flatten(ndim=2)
            f = theano.function([inputVariable], tempOut, allow_input_downcast=True)
            for name in  tempParams:
                tempParams[name].set_value(numpy.asarray(self.parameterMap[name], dtype='float32'),borrow=True)
            self.vectorFunction.append(f)

        # contructing word map
        self.singleWordVector = self.vectorFunction[0](numpy.asarray([range(len(self.wordVector))]).T)
#        self.singleWordVector = range(len(self.wordVector))
#        for i in range(len(self.singleWordVector)):
#            self.singleWordVector[i] = self.vectorFunction[0](numpy.asarray([[[self.wordVector[i]]]]))[0]
#        self.singleWordVector = numpy.asarray(self.singleWordVector)


        print "#### start testing ###"

        output = []
        inputWords = []
        try:
            while True:
                line = raw_input().decode(sys.stdin.encoding) 
                words = list(line)
                self.indexToWord.append(line)
                wordVector1 = self.getWordVectors(words)
                line = raw_input().decode(sys.stdin.encoding) 
                self.indexToWord.append(line)
                words = list(line)
                wordVector2 = self.getWordVectors(words)
                print getSim(wordVector1, wordVector2)

                self.singleWordVector = numpy.concatenate((self.singleWordVector, [wordVector1]))
                self.singleWordVector = numpy.concatenate((self.singleWordVector, [wordVector2]))
        except:
            outputFile = open('/home/ubuntu/outputVector', 'a')
            outputFile.write(cPickle.dumps(output))
            outputFile.close()

            # get top n similar
            line = raw_input().decode(sys.stdin.encoding) 
            words = list(line)
            wordVector = self.getWordVectors(words)

            records = []
            for index in range(len(self.singleWordVector)):
                pair = (getSim(wordVector, self.singleWordVector[index]), index)
                heapq.heappush(records, pair)
            largest = heapq.nlargest(11, records) 
            for index in largest:
                print self.indexToWord[index[1]], index[0]



            # get top n similar
#            records = []
#            for index in range(len(self.wordVector)):
#                pair = (getSim(wordVector, self.singleWordVector[index]), index)
#                heapq.heappush(records, pair)
#            largest = heapq.nlargest(11, records) 
#            for index in largest[1:]:
#                print self.indexToWord[index[1]], index[0]




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


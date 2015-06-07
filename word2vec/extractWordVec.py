#!/usr/bin/python
#coding=utf-8

# import modules & set up logging
import gensim, logging
import os
import sys
import multiprocessing
import codecs
import signal
import cPickle
import argparse

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Training word2vec.')
    parser.add_argument('-m', '--model', dest='model', help='the path of the pre-trained model')
    parser.add_argument('-o', '--output', dest='output', help='the path to store word embedding matrix')

    args = parser.parse_args()

    if (args.model == None or args.output == None ):
        parser.print_help()
        quit()


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    model = gensim.models.Word2Vec.load(args.model)
    
    output = {}
    output['index2word'] = model.index2word
    
    wordMatrix = []
    for word in model.index2word:
        wordMatrix.append(model[word])
    
    output['wordMatrix'] = wordMatrix
    
    outputFile = open(args.output,'w')
    outputFile.write(cPickle.dumps(output))
    print "dumps vector:", len(model.index2word)


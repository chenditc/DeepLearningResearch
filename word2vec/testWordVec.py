#!/usr/bin/python
#coding=utf-8

# import modules & set up logging
import gensim, logging
import os
import sys
import multiprocessing
import codecs
import signal
import argparse

if __name__ == "__main__" :

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Training word2vec.')
    parser.add_argument('-m', '--model', dest='model', help='the path of the pre-trained model')

    args = parser.parse_args()

    if (args.model == None):
        parser.print_help()
        quit()


    model = gensim.models.Word2Vec.load(args.model)
    
    while True:
        line = raw_input().decode(sys.stdin.encoding) 
        words = list(line)
        wordList =  model.most_similar(positive=words)
        for pair in wordList:
       	    print pair[0], pair[1]
    

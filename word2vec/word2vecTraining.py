#!/usr/bin/python
#coding=utf-8

# import modules & set up logging
import gensim, logging
import re
import os
import sys
import multiprocessing
import codecs
import signal
import argparse

import bigramerTraining 

def trainWord2Vec(inputDirectory, outputPath):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    class MySentences(object):
       def __init__(self, dirname):
           self.dirname = dirname
           self.numberHolder = re.compile(ur"[0-9]+", re.UNICODE)
           self.spliter = re.compile('\s+', re.UNICODE)

       def __iter__(self):
           for fname in os.listdir(self.dirname):
               for line in codecs.open(os.path.join(self.dirname, fname), encoding='utf-8'): 
                   line = re.sub(self.numberHolder, u' <num> ', line) # replace with space and <num> 
                   yield self.spliter.split(line)
    
    # handle sigterm
    def sigterm_handler(_signo, _stack_frame):
        # Raises SystemExit(0):
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    model = None
    try:
#        bigramer1 = gensim.models.Phrases(bigramerTraining.MySentences(inputDirectory), min_count=10000, threshold=0.1, delimiter='')
#        bigramer1.save(outputPath + "-bigramer")
#        sentences = bigramer1[bigramerTraining.MySentences(inputDirectory)] # a memory-friendly iterator
        sentences = MySentences(inputDirectory)
        model = gensim.models.Word2Vec(None, sg=1, size=200, window=10, min_count=10, workers=multiprocessing.cpu_count())
        model.build_vocab(sentences)
        sentences = gensim.utils.RepeatCorpusNTimes(sentences, 1)  # set iteration
        model.train(sentences)
        print model.most_similar(positive=[u'我'])
    finally:
        try:
            model.save(outputPath)
        except:
            print "Save model failed"

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Training word2vec.')
    parser.add_argument('-d', '--data', dest='data', help='the directory that store text data')
    parser.add_argument('-o', '--output', dest='output', help='the path to store model')

    args = parser.parse_args()

    if (args.data == None or args.output == None ):
        parser.print_help()
        quit()

    trainWord2Vec(args.data, args.output)


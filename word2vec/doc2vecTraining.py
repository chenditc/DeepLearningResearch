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

from gensim.models.doc2vec import LabeledSentence

def trainWord2Vec(inputDirectory, outputPath):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname
            self.spliter = re.compile(ur"[0-9]+|[a-z]+|.", re.UNICODE) 
            self.bigramer1 = gensim.models.Phrases(bigramerTraining.MySentences(inputDirectory), min_count=10000, threshold=0.1, delimiter='')

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in codecs.open(os.path.join(self.dirname, fname), encoding='utf-8'): 
                    yield LabeledSentence(words=self.bigramer1[self.spliter.findall(line)], labels=[fname])
 


    # handle sigterm
    def sigterm_handler(_signo, _stack_frame):
        # Raises SystemExit(0):
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    model = None
    try:
        sentences = MySentences(inputDirectory) # a memory-friendly iterator
        model = gensim.models.Doc2Vec(None, size=500, window=4, min_count=1, workers=multiprocessing.cpu_count())
        model.build_vocab(sentences)
        sentences = gensim.utils.RepeatCorpusNTimes(sentences, 10)  # set iteration
        model.train(sentences)
        print model.most_similar('sohu-2015-06-14-10:47')
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


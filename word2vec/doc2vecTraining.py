#!/usr/bin/python
#coding=utf-8

# import modules & set up logging
import gensim, logging
import re
import os
import sys
import multiprocessing
import signal
import codecs
import argparse

import bigramerTraining

from gensim.models.doc2vec import LabeledSentence

def trainWord2Vec(inputDirectory, outputPath):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname
#            self.spliter = re.compile(ur"[0-9]+|[a-z]+|.", re.UNICODE) 
#            print "Save bigram in: ", outputPath + "-bigramer"
#            self.bigramer1 = gensim.models.Phrases(bigramerTraining.MySentences(inputDirectory), min_count=10000, threshold=0.1, delimiter='')
#            self.bigramer1.save(outputPath + "-bigramer")

            self.numberHolder = re.compile(ur"[0-9]+", re.UNICODE)
            self.spliter = re.compile(r'\s+', re.UNICODE)

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                for line in codecs.open(os.path.join(self.dirname, fname), encoding='utf-8'): 
                    line = re.sub(self.numberHolder, u' <num> ', line) # replace with space and <num> 
                    # fetch ['2014', '05', '22']
                    dateElements = list(re.search(r'([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])', fname).groups())
                    # change '05' to '5'
                    date = '/'.join([str(int(dateNumber)) for dateNumber in dateElements])
                    yield LabeledSentence(words=self.spliter.split(line), labels=[date])

    # handle sigterm
    def sigterm_handler(_signo, _stack_frame):
        # Raises SystemExit(0):
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    model = None
    try:
        sentences = MySentences(inputDirectory) # a memory-friendly iterator
        model = gensim.models.Doc2Vec(None, size=500, window=4, min_count=5, workers=multiprocessing.cpu_count())
        model.build_vocab(sentences)
        sentences = gensim.utils.RepeatCorpusNTimes(sentences, 10)  # set iteration
        model.train(sentences)
#        for i in range(10):
#            model.train(sentences)
#            model.alpha -= 0.002
#            model.min_alpha = model.alpha
        print model.most_similar('2015/6/14')
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


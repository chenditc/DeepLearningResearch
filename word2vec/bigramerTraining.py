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

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.spliter = re.compile(ur"[0-9]+|[a-z]+|.", re.UNICODE) 

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in codecs.open(os.path.join(self.dirname, fname), encoding='utf-8'): 
                yield self.spliter.findall(line)
 
def trainBigram(inputDirectory, outputPath):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
   
    # handle sigterm
    def sigterm_handler(_signo, _stack_frame):
        # Raises SystemExit(0):
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    model = None
    try:
        bigramer1 = gensim.models.Phrases(MySentences(inputDirectory), min_count=10000, threshold=0.1, delimiter='')
        while True:
            line = raw_input().decode(sys.stdin.encoding) 
            words = list(line)
            print bigramer1(words)
    finally:
        try:
            bigramer1.save(outputPath)
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

    trainBigram(args.data, args.output)


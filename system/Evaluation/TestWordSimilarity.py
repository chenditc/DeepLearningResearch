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

print "import 1"
import numpy as np
print "import 1"

import matplotlib.pyplot as plt
print "import 1"

from matplotlib import offsetbox
print "import 1"

from sklearn.manifold import TSNE
print "import 1"


class TestModel():

    def __init__(self, data, warmupModel):
        # load word index mapping
        warmupModelString = open(warmupModel).read()
        self.indexToWord = cPickle.loads(warmupModelString)['index2word']
        self._wordToIndex = dict([(self.indexToWord[i], i) for i in range(len(self.indexToWord))])

        # load pre-trained word vector
        self.parameterMap = json.loads(open(data).read())
        self.wordVector = self.parameterMap['Projection']
        self.one_gram = self.parameterMap['Conv-1-Filter']
        

    def startTesting(self):
        model = TSNE(n_components=2, init='pca', random_state=0)
        print "start training"
        t0 = time.time()
        pictureMatrix = model.fit_transform(self.wordVector[:10])
        self.plot_embedding(pictureMatrix, "t-SNE embedding of the digits (time %.2fs)" %
                               (time.time() - t0))
        plt.show()
        plt.draw()
        plt.savefig('~/test.png', dpi=100)


    #----------------------------------------------------------------------
    # Scale and visualize the embedding vectors
    def plot_embedding(self, X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(digits.data.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)



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
    

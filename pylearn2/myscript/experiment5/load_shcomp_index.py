#!/usr/bin/python
import re
import os

import gensim
import numpy as np

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_security(security, start, stop, days=1):
    x = []
    y = []

    print "loading feature vector with news vector"

    lines = open(security).read().split('\n')
    ticker = lines[0]
    fields = str(lines[1])

    dataLines = lines[2:]

    # create financial feature map between date and feature
    featureMap = {}
    for i in range(len(dataLines) - days):
        date = re.split(r'\s', dataLines[i + days])[0]
        line = dataLines[i + days - 1]
        nextLine = dataLines[i + days]
        try:
            tempX = []
            tempY = []

            thisPrice = float(re.split(r'\s', line)[2])
            nextPrice = float(re.split(r'\s', nextLine)[2])
            # print date and prise in percentage
#            print re.split(r'\s', line)[0], (nextPrice - thisPrice) * 100.0 / thisPrice 
            # get the price change in percent
            tempY.append( (nextPrice - thisPrice) * 100.0 / thisPrice )

            for j in range(days):
                # split data points
                datapoints = re.split(r'\s', dataLines[i + j])[1:-1]
                for element in datapoints:
                    tempX.append(float(element))

            featureMap[date] = (tempX, tempY)
        except:
            continue

    # Read all news data and construct x and y
    newsDir = '/home/ubuntu/news_data/finance_news_data'
    newsFiles = os.listdir(newsDir)
    newsModel = gensim.models.doc2vec.Doc2Vec.load('./doc2vec.finance.model.6') 
    for fileName in newsFiles:
        if fileName not in newsModel.vocab:
            continue
        newsVector = newsModel[fileName]
        # fetch ['2014', '05', '22']
        dateElements = list(re.search(r'([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])', fileName).groups())
        # change '05' to '5'
        date = '/'.join([str(int(dateNumber)) for dateNumber in dateElements])
        if date not in featureMap:
            continue
        featureVector = newsVector.tolist() + featureMap[date][0]
        x.append(featureVector)
        y.append(featureMap[date][1])
        

    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)

    print "Feature Map:", len(featureMap)

    print len(x), len(y)

    x = x[start:stop, :] 
    y = y[start:stop, :]

    print "Dimension:", len(x[0])

    return DenseDesignMatrix(X=x, y=y)

if __name__ == "__main__":
    load_security('/home/ubuntu/SHCOMP_Index', 0, 3000, days=5)

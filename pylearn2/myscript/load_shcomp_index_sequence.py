#!/usr/bin/python
import numpy as np
import re

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_security(security, start, stop):
    x = []
    y = []

    lines = open(security).read().split('\n')
    ticker = lines[0]
    fields = str(lines[1])

    dataLines = lines[2:]
    for i in range(len(dataLines)-1):
        line = dataLines[i]
        nextLine = dataLines[i+1]
        try:
            # split data points
            datapoints = re.split(r'\s', line)[1:-1]
            thisPrice = float(re.split(r'\s', line)[2])
            nextPrice = float(re.split(r'\s', nextLine)[2])
            # print date and prise in percentage
#            print re.split(r'\s', line)[0], (nextPrice - thisPrice) * 100.0 / thisPrice 
            tempX = []
            tempY = []

            # get the price change in percent
            tempY.append( (nextPrice - thisPrice) * 100.0 / thisPrice )

            for element in datapoints:
                tempX.append(float(element))

            x.append(tempX)
            y.append(tempY)
        except:
            continue

    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)
    x = x[start:stop, :] 
    y = y[start:stop, :]

    return [x,y] 
if __name__ == "__main__":
    load_security('/home/ubuntu/SHCOMP_Index', 0, 3000)

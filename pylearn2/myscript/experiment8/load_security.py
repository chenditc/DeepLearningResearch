#!/usr/bin/python
import numpy as np
import re
import MySQLdb

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_index(start, stop, days=1):
    x = []
    y = []

    lines = open('/home/ubuntu/SHCOMP_Index').read().split('\n')
    # get rid of header
    dataLines = lines[2:]

    # create financial feature map between date and feature
    featureMap = {}
    for i in range(len(dataLines) - days):
        date = re.split(r'\s', dataLines[i + days - 1])[0]
        try:
            tempX = []
            # print date and prise in percentage
#            print re.split(r'\s', line)[0], (nextPrice - thisPrice) * 100.0 / thisPrice 
            # get the price change in percent
            for j in range(days):
                # split data points
                datapoints = re.split(r'\s', dataLines[i + j])[1:-1]
                for element in datapoints:
                    tempX.append(float(element))
            featureMap[date] = tempX
        except:
            continue
    return featureMap


def load_security(security, start, stop, days = 1):
    x = []
    y = []

    dbConnector = MySQLdb.connect(host="stockdb.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                        user="chenditc", 
                                        passwd="cd013001",
                                        db="ch_day_tech")
    cursor = dbConnector.cursor()
#    cursor.execute('SELECT * FROM data WHERE Ticker=%s', (security) )
    cursor.execute('SELECT * FROM data ORDER BY Ticker, Date LIMIT 700000')
    dataLines = cursor.fetchall()

    indexMap = load_index(start, stop, days)

    for i in range(len(dataLines)- days):
        # If ticker number is not same, skip
        if (dataLines[i][0] != dataLines[i + days][0]):
            continue

        tempX = []
        tempY = []

        # get y value
        thisLine = dataLines[i + days - 1]
        nextLine = dataLines[i + days]
        thisPrice = float(thisLine[3])
        nextPrice = float(nextLine[2])
        # get the price change in percent
        percentChange = (nextPrice - thisPrice) * 100.0 / thisPrice 
        tempY.append( percentChange )

        # get date
        date = str(thisLine[1].year) + '/' + str(thisLine[1].month) + '/' + str(thisLine[1].day)

        # parse x value
        shouldSkip = False
        for j in range(days):
            if 'N' in dataLines[i + j]:
                shouldSkip = True
                break
            # split data points
            datapoints = dataLines[i + j] 
            for element in datapoints[2:]:
                tempX.append(float(element))
        if shouldSkip:
            continue

        # append feature vectors
        if date not in indexMap:
            continue
        tempX += indexMap[date]

        x.append(tempX)
        y.append(tempY)

    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)
    x = x[start:stop, :] 
    y = y[start:stop, :]

    print "Length:", len(x), len(y)
    print "Dimension:", len(x[0])

    return DenseDesignMatrix(X=x, y=y)

if __name__ == "__main__":
    load_security('600004 CH Equity', 0, 80000, 10)

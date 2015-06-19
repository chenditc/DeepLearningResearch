#!/usr/bin/python
import numpy as np
import re
import MySQLdb

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_security(security, start, stop):
    x = []
    y = []

    dbConnector = MySQLdb.connect(host="stockdb.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                        user="chenditc", 
                                        passwd="cd013001",
                                        db="ch_day_tech")
    cursor = dbConnector.cursor()
#    cursor.execute('SELECT * FROM data WHERE Ticker=%s', (security) )
    cursor.execute('SELECT * FROM data LIMIT 100000')
    dataLines = cursor.fetchall()

    for i in range(len(dataLines)-1):
        line = dataLines[i]
        nextLine = dataLines[i+1]
        try:
            # split data points
            datapoints = line 
            thisPrice = float(line[3])
            nextPrice = float(nextLine[2])
            # print date and prise in percentage
#            print line[1], (nextPrice - thisPrice) * 100.0 / thisPrice 
            tempX = []
            tempY = []

            # get the price change in percent
            tempY.append( (nextPrice - thisPrice) * 100.0 / thisPrice )

            for element in datapoints[2:]:
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


    return DenseDesignMatrix(X=x, y=y)

if __name__ == "__main__":
    load_security('600004 CH Equity', 0, 80000)

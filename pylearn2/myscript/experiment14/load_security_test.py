#!/usr/bin/python
import re
import time
import json
import sys

import numpy as np
import MySQLdb

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

dbConnector = MySQLdb.connect(host="stockdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                    user="chenditc", 
                                    passwd="cd013001",
                                    db="ch_day_tech")
newsMatrixMap = {}

def getNewsAtDate(date):
    if date in newsMatrixMap:
        return newsMatrixMap[date]

    cursor = dbConnector.cursor()

    # Get all important date
    sql = '''
SELECT news_vector FROM ch_day_tech.news_vec WHERE date = '{0}'
'''
    cursor.execute(sql.format(date))
    news = cursor.fetchall()
    newsMatrix = []
    for oneNews in news:
        newsMatrix.append(json.loads(oneNews[0]))
    newsMatrixMap[date] = newsMatrix
    return newsMatrix 


def getImportantDates(security = 'all', minDate = '1970-01-01', maxDate = '2070-01-01', threshold = 5):
    cursor = dbConnector.cursor()

    securityFilter = ""
    if security != 'all':
        securityFilter = "and stock.Ticker = '" + security + "'" 

    # Get all important date
    sql = '''
SELECT original.Ticker, date_sub(original.date, INTERVAL 1 DAY) as date, important.px_change
FROM ch_day_tech.data as original,
(SELECT Ticker, Date, ((stock.PX_LAST / stock.PX_OPEN) - 1) * 100 as px_change 
    FROM ch_day_tech.data as stock,
    (SELECT min(date) as news_date 
        FROM ch_day_tech.news_vec ) as news
    WHERE stock.date < '{0}' 
    and stock.date > news.news_date
    and stock.date > '{1}'
    {3}
    and (stock.PX_LAST > stock.PX_OPEN * {2}
        or stock.PX_LAST < stock.PX_OPEN * {4}) ) as important
WHERE original.Ticker = important.Ticker 
and original.Date = important.Date;
    '''
    cursor.execute(sql.format(maxDate, minDate, str(1.0 + threshold / 100.0), securityFilter, str(1.0 - threshold / 100.0) ))
    importantDates = cursor.fetchall()
    return importantDates

def getStockData(security, date, days=5):
    sql = '''
SELECT index_vec.*, security_vec.* 
FROM 
(SELECT * FROM ch_day_tech.data 
    WHERE date <= '{0}' 
    and Ticker = 'SHCOMP Index' 
    ORDER BY Ticker, date desc) as index_vec,
(SELECT * FROM ch_day_tech.data 
    WHERE date <= '{0}' 
    and Ticker = '{1}' 
    ORDER BY Ticker, date desc) as security_vec 
WHERE index_vec.date = security_vec.date LIMIT {2};
'''
    cursor = dbConnector.cursor()
    cursor.execute(sql.format(date, security, str(days)))
    lines = cursor.fetchall()
    data = []
    for line in lines:
        data += [x for x in line if isinstance(x, float)]
    return data
 
def appendNewsData(stockVector, date):
    resultMatrix = []
    newsVectorList = getNewsAtDate(date) 
    for newsVector in newsVectorList:
        resultMatrix.append(newsVector + stockVector)
    return resultMatrix

def load_security(security = 'all', startDate = '1970-01-01', stopDate = '2070-01-01', days = 5, threshold = 5):
    print "loading data:", security, startDate, stopDate, days
    sys.stdout.flush()

    x = []
    y = []

    importantDates = []
    importantDates += list(getImportantDates(security = security, minDate = startDate, maxDate = stopDate, threshold = threshold))
    stockData = []
    for i in range(len(importantDates)):
        if i % 100 == 99:
            print "loading stock data:", i
        ticker = importantDates[i][0]
        date = importantDates[i][1]
        change = importantDates[i][2]
        stockVector = getStockData(security = ticker, date = date, days = days)
        if len(stockVector) != 88 * days:
            print "skipping stock data:", ticker, date
            continue
        tempX = appendNewsData(stockVector, date) 
        tempY = [change] * len(tempX)
        x += tempX
        y += tempY

    x = np.asarray(x)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)

    print "Length:", len(x), len(y)
    print "Dimension:", len(x[0])
    print "x shape: ", x.shape
    print "y shape: ", y.shape
    assert(x.shape[0] > 0)
    assert(y.shape[0] > 0)

    sys.stdout.flush()

    return DenseDesignMatrix(X=x, y=y)

if __name__ == "__main__":
    load_security('all', startDate =  '2013-01-01',  stopDate = '2013-03-01', days = 5)
#    load_security(None, '2013-01-01', '2014-01-01', 5)

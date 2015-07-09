#!/usr/bin/python -u 
import re
import time
import json
import sys
import multiprocessing
from multiprocessing import Pool

import numpy as np
import MySQLdb
from pylearn2.space import CompositeSpace
from pylearn2.space import VectorSpace
from pylearn2.space import IndexSpace

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset

dbConnector = MySQLdb.connect(host="stockdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                    user="chenditc", 
                                    passwd="cd013001",
                                    db="ch_day_tech")
dbConnectorList = []
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


def getImportantDates(security = 'all', minDate = '1970-01-01', maxDate = '2070-01-01', threshold = 5 , limit = 0):
    cursor = dbConnector.cursor()

    securityFilter = ""
    if security != 'all':
        securityFilter = "and stock.Ticker = '" + security + "'" 

    limitFilter = ""
    if limit != 0:
        limitFilter = " LIMIT " + str(limit)

    importantDates = []

    # Get positive important date
    sql = '''
SELECT original.Ticker, date_sub(original.date, INTERVAL 1 DAY) as date, important.px_change
FROM ch_day_tech.data as original,
(SELECT Ticker, Date, ((stock.PX_LAST / stock.PX_OPEN) - 1) * 100 as px_change 
    FROM ch_day_tech.data as stock,
    (SELECT min(date) as news_date 
        FROM ch_day_tech.news_vec ) as news
    WHERE stock.date < '{0}' 
    and stock.date > '{1}'
    {3}
    and stock.PX_LAST > stock.PX_OPEN * {2} ) as important
WHERE original.Ticker = important.Ticker 
and original.Date = important.Date {4};
    '''
    cursor.execute(sql.format(maxDate, minDate, str(1.0 + threshold / 100.0), securityFilter, limitFilter))
    positiveDates = list(cursor.fetchall())

    # Get positive important date
    sql = '''
SELECT original.Ticker, date_sub(original.date, INTERVAL 1 DAY) as date, important.px_change
FROM ch_day_tech.data as original,
(SELECT Ticker, Date, ((stock.PX_LAST / stock.PX_OPEN) - 1) * 100 as px_change 
    FROM ch_day_tech.data as stock,
    (SELECT min(date) as news_date 
        FROM ch_day_tech.news_vec ) as news
    WHERE stock.date < '{0}' 
    and stock.date > '{1}'
    {3}
    and (stock.PX_LAST < stock.PX_OPEN * {2}) ) as important
WHERE original.Ticker = important.Ticker 
and original.Date = important.Date {4};
    '''
    cursor.execute(sql.format(maxDate, minDate, str(1.0 - threshold / 100.0), securityFilter, limitFilter))
    negativeDates = list(cursor.fetchall())

    dataLength = min(len(positiveDates), len(negativeDates)) 
    importantDates = positiveDates[:dataLength] + negativeDates[:dataLength]
    print "data for threshold:", threshold, " length:", dataLength * 2
    return importantDates

def getStockData(inputTuple):
    security, date, changes = inputTuple
    days = 5

    print "loading stock data:", security, date
    sql = '''
SELECT security_vec.*, index_vec.* 
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
    stockdbConnector = MySQLdb.connect(host="stockdb2.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                        user="chenditc", 
                                        passwd="cd013001",
                                        db="ch_day_tech")
    cursor = stockdbConnector.cursor()
    cursor.execute(sql.format(date, security, str(days)))
    lines = cursor.fetchall()
    data = []
    for line in lines:
        data += [x for x in line if isinstance(x, float)]
    return (security, date, changes, data)
 
def appendNewsData(stockVector, date):
    resultMatrix = []
    newsVectorList = getNewsAtDate(date) 
    for newsVector in newsVectorList:
        resultMatrix.append(newsVector + stockVector)
    return resultMatrix

def getTickerToIndex():
    sql = '''
SELECT DISTINCT(Ticker) FROM ch_day_tech.data;
'''
    cursor = dbConnector.cursor()
    cursor.execute(sql)
    lines = cursor.fetchall()
    data = {}
    for i in range(len(lines)):
        data[lines[i][0]] = i
    return data


def load_security(security = 'all', startDate = '1970-01-01', stopDate = '2070-01-01', days = 5, threshold = 5):
    print "loading data:", security, startDate, stopDate, days
    sys.stdout.flush()

    x = []
    y = []

    importantDates = []
    for i in range(threshold, 20):
        importantDates += list(getImportantDates(security = security, minDate = startDate, maxDate = stopDate, threshold = i))
    normalDates = list(getImportantDates(security = security, minDate = startDate, maxDate = stopDate, threshold = 0))
    importantDates = importantDates * (len(normalDates) / len(importantDates))
    importantDates += normalDates

    tickerToIndex = getTickerToIndex()

    print "about to load ", len(importantDates)
    jobList = []
    for i in range(len(importantDates)):
        ticker = importantDates[i][0]
        date = importantDates[i][1]
        change = importantDates[i][2]
        jobList.append((ticker, date, change))

    # TODO: fetch data with multiprocess
    workerNumber = 200
    workerPool = Pool(workerNumber)
    dataList = workerPool.map(getStockData, jobList)
    print "Finish loading data"

    tickerIndex = []
    stockData = []
    changeArray = []
    for dataTuple in dataList:
        ticker, date, change, stockVector = dataTuple
        # validate data
        if len(stockVector) != 88 * days:
            print "skipping stock data:", ticker, date
            continue
        tickerIndex.append([tickerToIndex[ticker]])
        stockData += [stockVector]
        changeArray += [change]

    tickerIndex = np.asarray(tickerIndex)
    stockData = np.asarray(stockData)
    changeArray = np.asarray(changeArray)
    changeArray = changeArray.reshape(changeArray.shape[0], 1)
    data = (tickerIndex, stockData, changeArray)

    print "Length:", len(tickerIndex), len(stockData), len(changeArray)
    print "Dimension:", len(stockData[0])
    assert(stockData.shape[0] > 0)
    assert(changeArray.shape[0] > 0)
    assert(tickerIndex.shape[0] > 0)
    sys.stdout.flush()

    # define the data specs
    space = CompositeSpace([
        IndexSpace(dim=1, max_labels = 2000),      # ticker index
        VectorSpace(dim=440),   # stock 
        VectorSpace(dim=1)])    # target
    source = ('features0','features1','targets')
    data_specs = (space,source)
    return VectorSpacesDataset(data=data,data_specs=data_specs)

if __name__ == "__main__":
    load_security('all', startDate =  '2013-02-01',  stopDate = '2013-03-01', days = 5)
#    load_security(None, '2013-01-01', '2014-01-01', 5)

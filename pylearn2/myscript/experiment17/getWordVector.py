
from pylearn2.utils import serial
import numpy
import load_security_test
import MySQLdb
import theano
import heapq
import datetime
import operator
from scipy.linalg import norm

dbConnector = MySQLdb.connect(host="stockdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                    user="chenditc", 
                                    passwd="cd013001",
                                    db="ch_day_tech")

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


model_path = './mlp_regression1_best.pkl'
model = serial.load( model_path )

X = theano.tensor.imatrix()
print X
Y = model.layers[0].layers[0].fprop( X )
print Y


#X = model.get_input_space().make_theano_batch()

predictFunction = theano.function( [X], Y ,allow_input_downcast=True)

tickerToIndex = getTickerToIndex()
inputMatrix = []
for i in range(1999):
    inputMatrix.append([i])

stockMatrix = predictFunction(inputMatrix)
vector1 = stockMatrix[tickerToIndex['600108 CH Equity']]

similarityArray = []
for i in range(1999):
    score = numpy.dot(vector1, stockMatrix[i]) / (norm(vector1) * norm(stockMatrix[i]))
    similarityArray.append(score)

printPair = []
for stock in tickerToIndex:
    printPair.append( (stock, similarityArray[tickerToIndex[stock]]) )
printPair.sort(key=lambda x:x[1])

for pair in printPair:
    print pair



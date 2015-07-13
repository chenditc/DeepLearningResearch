
from pylearn2.utils import serial
import numpy
import load_security_test
import theano
import heapq
import datetime
import operator

model_path = './mlp_regression1_best.pkl'
model = serial.load( model_path )
X = model.get_input_space().make_theano_batch()
Y = model.fprop( X )
predictFunction = theano.function( list(X), Y ,allow_input_downcast=True)

myMoney = 100

for dateDelta in range(300):
    today = datetime.date(2014, 3, 5) + datetime.timedelta(dateDelta)

    if today.weekday() not in range(5):
        continue

    startDate = (today + datetime.timedelta(-1)).strftime("%y-%m-%d")
    stopDate = (today + datetime.timedelta(1)).strftime("%y-%m-%d")
    loadDataReturn = load_security_test.load_security('all', startDate = startDate, stopDate = stopDate, days = 5, threshold = 0)
    if loadDataReturn == None:
        continue
    dataset, indexToSecuirty = loadDataReturn
    x_test = list(dataset.data)[:2]

    y = predictFunction( x_test[0], x_test[1] )

    # get all security - score pair
    aggregatedResult = {}
    for i in range(len(indexToSecuirty)):
        aggregatedResult[indexToSecuirty[i]] = y[i]

    scoreHeap = sorted(aggregatedResult.items(), key=operator.itemgetter(1), reverse=True)

    pnls = []
    for i in range(5):
        invest = scoreHeap[i] 
        security = invest[0]
        predictChange = invest[1]
        realChange =  dataset.data[2][indexToSecuirty.index(security)]
        pnl = 0
        if predictChange > 0:
            pnl += realChange
        else:
            pnl -= realChange
        pnls += [pnl]
        print "For ", security, " predicti:", predictChange, " actual:", realChange, " P&L:", pnl 

    for i in range(5):
        invest = scoreHeap[-1-i] 
        security = invest[0]
        predictChange = invest[1]
        realChange =  dataset.data[2][indexToSecuirty.index(security)]
        pnl = 0
        pnl -= realChange
        pnls += [pnl]
        print "For ", security, " predicti:", predictChange, " actual:", realChange, " P&L:", pnl 

    myMoney *= (1 + numpy.average(pnls) / 100.0) 
    print "My money:", myMoney



#!/usr/bin/python
import random
import numpy


# Get x Label
def getRandomSeries():
    x = [0]
    for i in range(0, 50000):
        xLast = x[len(x)-1]
        xNew = xLast + random.uniform(-1,1)
        x.append(xNew)
    xMax = max(x)
    xMin = min(x)
    result = []
    for i in x:
        xNew = (i - xMin) / (xMax - xMin)
        result.append(xNew)
    return result

# get y label
# y = 1 if x1 cross x2 from downward
# y = -1 if x1 cross x2 from upward
# y = 0  they don't cross
def getCrossLabel(x1, x2):
    assert(len(x1) == len(x2))
    length = len(x1)

    y = [0]
    for i in range(1, length):
        x10 = x1[i-1]
        x20 = x2[i-1]
        x11 = x1[i]
        x21 = x2[i]
        yNew = 0
        if (x10 - x20 < 0 and x11 - x21 > 0):
            yNew = 1
        if (x10 - x20 > 0 and x11 -x21 < 0):
            yNew = 2
        y.append(yNew)

    assert(len(x1) == len(y))
    return y

def getMovingAverage(timeSeries, day):
    # padding 0 for initial values
    y = [0] * day
    for i in range(day, len(timeSeries)): 
        sumOfDays = sum(timeSeries[i-day+1:i+1])
        avg = sumOfDays / day
        y.append(avg)
    return y
        
x = getRandomSeries()
x1 = getMovingAverage(x, 3) 
x2 = getMovingAverage(x, 6)
y = getCrossLabel(x1, x2)

for i in range(0, len(y)):
#    mean = numpy.mean([x1[i-1], x1[i], x2[i-1], x2[i]])
    mean = 0
    row = [x1[i-1] - mean , x1[i] - mean, x2[i-1] - mean,  x2[i] - mean, y[i]]
#    row[0:3] = [x * 20 for x in row[0:3]] 
    temp = ",".join(str(x) for x in row)
    print temp

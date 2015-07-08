#!/usr/bin/python

import numpy

lines = open('temp').read().split('\n')
lines.remove('')
myMoney = [float(line.split(' ')[2]) for line in lines]
changes = []
for i in range(len(myMoney) - 1):
    changes.append(myMoney[i+1] - myMoney[i])

print "item" , len(changes)
print "max", max(changes)
print "min", min(changes)
print "wrong: ", len([x for x in changes if x < 0])
print "right: ", len([x for x in changes if x > 0])
print "std:", numpy.std(changes)
print "avg", numpy.average(changes)
print "sharpe", numpy.average(changes) / numpy.std(changes)
print "result", myMoney[-1] 

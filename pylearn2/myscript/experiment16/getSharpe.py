import numpy

lines = open('temp').read().split('\n')
lines.remove('')
lines = [line.split(' ') for line in lines]
checkMap = {}
changeMap = {}
predictMap = {}
for line in lines:
    if line[3] == 'True':
        checkMap[line[1]] = True 
    else:
        checkMap[line[1]] = False
    changeMap[line[1]] = float(line[1])
    predictMap[line[1]] = abs(float(line[0]))

changes = []
myMoney = 100
buyhold = 100
upCount = 0
for key in checkMap:
    if predictMap[key] < 3:
        continue
    if checkMap[key]:
        changes.append(abs(changeMap[key]))
        myMoney *= 1 + abs(changeMap[key]) / 100
    else:
        changes.append(0 - abs(changeMap[key]))
        myMoney *= 1 - abs(changeMap[key]) / 100

    buyhold *= 1 + changeMap[key] / 100

    if changeMap[key] > 0:
        upCount += 1


print "item" , len(changes)
print "max", max(changes)
print "min", min(changes)
print "wrong: ", len([x for x in changes if x < 0])
print "right: ", len([x for x in changes if x > 0])
print "baseRight: ", upCount
print "std:", numpy.std(changes)
print "avg", numpy.average(changes)
print "sharpe", numpy.average(changes) / numpy.std(changes)
print "result", myMoney 
print "baseline", buyhold

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
for key in checkMap:
    if predictMap[key] < 6:
        continue
    if checkMap[key]:
        changes.append(abs(changeMap[key]))
    else:
        changes.append(0 - abs(changeMap[key]))

print len(changes)
print max(changes)
print min(changes)
print "wrong: ", len([x for x in changes if x < 0])
print "right: ", len([x for x in changes if x > 0])
print "std:", numpy.std(changes)
print "avg", numpy.average(changes)
print "sharpe", numpy.average(changes) / numpy.std(changes)
myMoney = 100
for x in changes:
    myMoney *= 1 + x / 100
print "result", myMoney 

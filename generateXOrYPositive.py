import random

# y = 0 if x1*x2 < 0
# y = 1 if x1 > 0 and x2 > 0
# y = 2 if x1 < 0 and x2 < 0

# get a random int series

total = 10000

x = []
y = []
for i in range(0, total):
    x.append(random.uniform(-10,10));
    if (x[i-1]*x[i] < 0):
        y.append(0)
    elif x[i-1] > 0 and x[i] > 0:
        y.append(1)
    else:
        y.append(2)

# y label is the sum of previous 2 number
for i in  range(0, total):
    print x[i-1], ",",  x[i], ",", y[i]

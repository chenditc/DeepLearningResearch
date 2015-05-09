import random

# Generate data where sum of x1 and x2 is positive or not. Simple test for logistic regression model.

# get a random int series

total = 10000

x = []
y = []
for i in range(0, total):
    x.append(random.uniform(-10,10));
    if (x[i-1] + x[i] > 0):
        y.append(1)
    else:
        y.append(0)

# y label is the sum of previous 2 number
for i in  range(0, total):
    print x[i-1], ",",  x[i], ",", y[i]

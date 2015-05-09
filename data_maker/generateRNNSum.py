import random

a = []
b = []
for i in range(50000):
    a.append(random.randint(0,100))

for i in range(50000):
    if a[i] + a[i-1] + a[i-2] > 150:
        b.append(1)
    else:
        b.append(0)

for i in range(50000):
    print a[i], "," , b[i]

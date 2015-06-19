
from pylearn2.utils import serial
import load_shcomp_index
import theano

dataset = load_shcomp_index.load_security('/home/ubuntu/SHCOMP_Index', start = 40000, stop=61000, days = 10)
#dataset = load_shcomp_index.load_security('/home/ubuntu/test.txt', start = 0, stop=130, days = 5)

x_test = dataset.X

model_path = './mlp_regression1_best.pkl'
model = serial.load( model_path )
X = model.get_input_space().make_theano_batch()
Y = model.fprop( X )

f = theano.function( [X], Y ,allow_input_downcast=True)

y = f( x_test )

right = 0
wrong = 0
myMoney = 100.0
standard = 100.0
for i in range(len(y)):
    signCorrect = False 
    up = True
    if dataset.y[i][0] < 0:
        up = False
    
    before = myMoney
    if y[i][0] > 0:
        myMoney += myMoney * dataset.y[i][0] / 100.0
    else:
        myMoney -= myMoney * dataset.y[i][0] / 100.0

    print myMoney, "Change:", (myMoney - before) / before
    standard += standard * dataset.y[i][0] / 100.0

    if y[i][0] * dataset.y[i][0] > 0:
        signCorrect = True
        right += 1
    else:
        wrong += 1

    print y[i][0], dataset.y[i][0], up, signCorrect

print "Right: ", right, " Wrong: ", wrong
print "MyMoney: ", myMoney, " standard: ", standard 
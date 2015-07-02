
from pylearn2.utils import serial
import load_security
import theano

dataset = load_security.load_security('all', startDate = '2014-11-01',  days = 5, threshold = 0)
#dataset = load_shcomp_index.load_security('/home/ubuntu/test.txt', start = 0, stop=130, days = 5)

x_test = dataset.X

model_path = './mlp_regression1.pkl'
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
    threshold = 0
    if y[i][0] > threshold:
        myMoney += myMoney * dataset.y[i][0] / 100.0
    elif y[i][0] < -threshold:
        myMoney -= myMoney * dataset.y[i][0] / 100.0

    standard += standard * dataset.y[i][0] / 100.0

    if (myMoney - before) / before > 0:
        signCorrect = True
        right += 1
    elif (myMoney - before) / before < 0:
        wrong += 1
    else:
        continue

    print myMoney, "Change:", (myMoney - before) / before
    print y[i][0], dataset.y[i][0], up, signCorrect

print "Right: ", right, " Wrong: ", wrong
print "MyMoney: ", myMoney, " standard: ", standard 

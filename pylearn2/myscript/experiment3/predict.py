
from pylearn2.utils import serial
import load_shcomp_index
import theano

dataset = load_shcomp_index.load_security('/home/ubuntu/SHCOMP_Index', start = 3000, stop=5800, days = 5)
x_test = dataset.X

model_path = '/home/ubuntu/DeepLearningResearch/pylearn2/myscript/experiment3/mlp_regression1_best.pkl'
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
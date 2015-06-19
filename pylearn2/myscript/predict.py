
from pylearn2.utils import serial
import load_shcomp_index
import theano

dataset = load_shcomp_index.load_security('/home/ubuntu/SHCOMP_Index', start = 3000, stop=5000)
x_test = dataset.X

model_path = '/home/ubuntu/DeepLearningResearch/pylearn2/myscript/mlp_regression2.pkl'
model = serial.load( model_path )
X = model.get_input_space().make_theano_batch()
Y = model.fprop( X )

f = theano.function( [X], Y ,allow_input_downcast=True)

y = f( x_test )


for i in range(len(y)):
    signCorrect = False 
    up = True
    if dataset.y[i] < 0:
        up = False

    if y[i] * dataset.y[i] > 0:
        signCorrect = True

    print y[i], dataset.y[i], up, signCorrect
    

import theano.tensor as T
import theano

import LossFunctions
import Model

##
# @brief    This class is a virtual class that contain few method that will be share among classifiers
class Classifier(Model.Model):

    def __init__(self):
        print "Initializing Classifier"
        self.getTestError = self.getClassificationError

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        self._x = T.matrix('x')  # data, presented as rasterized images
        self._y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        # initialize model
        self._trainModel = None
        self._testModel = None

    ##
    # @brief                Given a list of inputs
    #
    # @param testInput      test input
    # @param testOutput     right output
    #
    # @return               error rate 
    def getClassificationError(self, testInput, testOutput):
        # compare the prediction with the output
        error = 0.0
        prediction = self.testModel(testInput)
        testOutput = testOutput.eval()
        for i in range(len(prediction)):
            if prediction[i] != testOutput[i]:
                error += 1

        return error / len(prediction)


    ##
    # @brief                Create theano function that take training x and y
    #                       and update W and b
    #
    # @param train_set_x    
    # @param train_set_y
    # @param parameterToTrain   A list of parameter index to train in this training model
    # @param lossFunction       The loss function that will calculate gradient. A function takes x and y
    # @param learning_rate      the step updateing parameters
    # @param batch_size         The size of training data to compute at once
    #
    # @return 
    def buildTrainingModel(self, 
                           train_set_x, 
                           train_set_y,
                           parameterToTrain = [],
                           lossFunction = LossFunctions.LossFunctions.negative_log_likelihood, 
                           learning_rate = 0.1, 
                           batch_size = 20):

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        index = T.lscalar()  # index to a [mini]batch

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = lossFunction(self.p_y_given_x, self._y)

        #TODO: make regularization configurable
        L2_reg=0.0001
        L1_reg=0.001
        L1 = sum([abs(self.params[key]).sum() for key in self.params])
        L2 = sum([(self.params[key] ** 2).sum() for key in self.params])
        cost = cost + L1_reg * L1 + L2_reg * L2 

        updates = self.getUpdateForVariable(cost, learning_rate, self.params, onlyTrain=parameterToTrain)

        # compiling a Theano function `trainModel` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self._trainModel = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self._x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self._y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self._totalBatches = train_set_x.get_value(borrow=True).shape[0] / batch_size


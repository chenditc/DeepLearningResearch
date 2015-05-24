import theano.tensor as T
import theano

import LossFunctions
import Model

##
# @brief    This class is a virtual class that contain few method that will be share among classifiers
class Classifier(Model.Model):

    def __init__(self):
        super(Classifier, self).__init__()
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
#        testOutput = testOutput.eval()
        for i in range(len(testOutput)):
            if prediction[i] != testOutput[i]:
                error += 1

        return error / len(prediction)


    ##
    # @brief                Create theano function that take training x and y
    #                       and update W and b
    #
    # @param lossFunction       The loss function that will calculate gradient. A function takes x and y
    # @param learning_rate      the step updateing parameters
    # @param batch_size         The size of training data to compute at once
    #
    # @return 
    def buildTrainingModel(self, 
                           lossFunction = LossFunctions.LossFunctions.negative_log_likelihood):

        ######################
        # BUILD ACTUAL MODEL #
        ######################

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = lossFunction(self.p_y_given_x, self._y)

        #TODO: make regularization configurable
        L2_reg=0.0001
        L1_reg=0.00
        L1 = sum([abs(self.params[key]).sum() for key in self.params])
        L2 = sum([(self.params[key] ** 2).sum() for key in self.params])
        cost = cost + L1_reg * L1 + L2_reg * L2 

        self.gradientsName, gradients = self.getGradientForVariable(cost)

        self._trainModel = theano.function(
            inputs=[self._x, self._y],
            outputs=gradients,
            allow_input_downcast=True
        )


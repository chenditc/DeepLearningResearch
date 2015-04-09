import Model

##
# @brief    This class is a virtual class that contain few method that will be share among classifiers
class Classifier(Model.Model):

    def __init__(self):
        print "Initializing Classifier"
        self.getTestError = self.getClassificationError

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


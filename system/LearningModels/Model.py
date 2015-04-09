import theano


class NoImplementationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("No Implementation: " + json.dumps(self.value) )



##
# @brief abstract class for machine learning models
#
class Model(object):
    ##
    # @brief                Given a list of inputs
    #
    # @param testInput      test input
    # @param testOutput     right output
    #
    # @return               error rate 
    def getTestError(self, testInput, testOutput):
        raise NoImplementationError("getTestError") 

    ##
    # @brief            Given a list of inputs , return a list of prediction
    #
    # @param testInput  
    #
    # @return 
    def testModel(self, testInput):
        # build test Model
        if (isinstance(testInput, theano.compile.sharedvalue.SharedVariable)):
            testInput = testInput.get_value()

        # loop through the input and compute prediction
        preditction = self._testModel(testInput)
        return preditction 


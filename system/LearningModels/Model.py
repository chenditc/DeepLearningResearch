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


    ##
    # @brief                Given a json string which wrap all the parameters,
    #                       Initialize the model to a trained state
    #
    # @param jsonString     A json string contains a map, key is the variable name
    #                       Value is the corresponding value of the variable
    #
    # @return 
    def loadModelFromJson(self, jsonString):
        raise NoImplementationError("loadModelFromJson")

import theano
import theano.tensor as T
import json
import numpy

import DataLoader


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
        parameters = json.loads(jsonString)
        for key in parameters:
            self.params[key] = theano.shared(
                value=numpy.asarray(
                    parameters[key],
                    dtype=theano.config.floatX
                ),
                name=key,
                borrow=True
            )


    ##
    # @brief        store the parameters to a json string
    #
    # @return       json string that represent the parameter map
    def storeModelToJson(self):
        parameters = {}
        for key in self.params:
            parameters[key] = self.params[key].get_value().tolist()
        return json.dumps(parameters)

    ##
    # @brief                Upload necessary data to the database
    #
    # @param dataLoader
    #
    # @return 
    def uploadModel(self ,dataLoader, errorRate):
        cursor = dataLoader.getDatabaseCursor()
        parameters = self.storeModelToJson()
        cursor.execute('INSERT INTO Model1 ( model_name, parameters, description, data_id, error_rate ) VALUES (%s, %s, %s, %s, %s)',
                        (self.__class__.__name__, parameters, self.__class__.description, dataLoader._data_id, errorRate))
        dataLoader.commitData()

    ##
    # @brief                Get update array from learning rate, parameters and etc.
    #
    # @param cost           The cost function to minimize
    # @param learningRate   
    # @param parameters     The tensor variable of all parameters in the model
    # @param onlyTrain      The parameters that will be trained. If the array is empty, update all
    #
    # @return               the array of update parameters 
    def getUpdateForVariable(self, cost, learningRate, parameters, onlyTrain=[]):
        # compute the gradient of cost with respect to theta = (W,b)
        if len(onlyTrain) == 0:
            onlyTrain = parameters.keys()

        updates = []
        for key in onlyTrain:
            gradient = T.grad(cost=cost, wrt=parameters[key])
            # specify how to update the parameters of the model as a list of
            # (variable, update expression) pairs.
            updates.append(
                    (parameters[key], parameters[key] - learningRate * gradient)
            )
        return updates



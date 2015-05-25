import theano
import theano.tensor as T
import json
import numpy
import redis

import DataLoader


class NoImplementationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("No Implementation: " + json.dumps(self.value) )



class Model(object):

    def __init__(self):
        self.redisClient = redis.StrictRedis(host='deeplearning.qha7wz.ng.0001.usw2.cache.amazonaws.com', port=6379, db=0)

    @staticmethod
    def loadModelByName(name, n_in, n_out):
        module = __import__(name)
        className = getattr(module, name)
        # The dimension here should be dummy, does not influence the model
        classInstance = className(n_in, n_out)
        return classInstance


    def trainModel(self, x, y):
        self.downloadModel()
        gradients = self._trainModel(x, y)
        return self.gradientsName, gradients

    ##
    # @brief                Download the paramters from database and set them in shared variable 
    #
    # @param dataLoader
    #
    # @return 
    def downloadModel(self):
        oldParameters = self.storeModelToJson() 
        for key in self.params:
            jsonString = self.redisClient.get(key)
            if jsonString == None:
                self.redisClient.set(key, oldParameters[key])
                continue

            self.loadParameterFromJson(key, jsonString)

    def loadParameterFromJson(self, jsonKey, jsonString):
        parameter = json.loads(jsonString)
        self.params[jsonKey].set_value(parameter)


    ##
    # @brief        store the parameters to a json string
    #
    # @return       json string that represent the parameter map
    def storeModelToJson(self):
        parameters = {}
        for key in self.params:
            parameters[key] = json.dumps(self.params[key].get_value().tolist())
        return parameters

    ##
    # @brief                Upload necessary data to the database
    #
    # @param dataLoader     the loader will be used to upload data
    # @param errorRate      error rate of database
    #
    # @return 
    def uploadModel(self ,dataLoader, errorRate):
        parameters = self.storeModelToJson()
        for key in parameters:
            self.redisClient.set(key, parameters[key])


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
        gradients = {}
        for key in onlyTrain:
            gradient = T.grad(cost=cost, wrt=parameters[key])
            gradients[key] = gradient

        # Get the norm of the gradients
        norm = T.sqrt(T.sum(
            [T.sum(param_gradient ** 2) for param_gradient in gradients.values()]
            ))
        for key in onlyTrain:
            gradients[key] = T.switch(
                    T.ge(norm, 1),             # 1 is the clipping value
                    gradients[key] / norm * 1,       # TODO: change it to a variable
                    gradients[key]
            )

        for key in onlyTrain:
            # specify how to update the parameters of the model as a list of
            # (variable, update expression) pairs.
            updates.append(
                    (parameters[key], parameters[key] - learningRate * gradients[key])
            )
        return updates


    ##
    # @brief                Get update array from learning rate, parameters and etc.
    #
    # @param cost           The cost function to minimize
    # @param learningRate   
    # @param parameters     The tensor variable of all parameters in the model
    # @param onlyTrain      The parameters that will be trained. If the array is empty, update all
    #
    # @return               the array of update parameters 
    def getGradientForVariable(self, cost):
        # compute the gradient of cost with respect to theta = (W,b)
        parameters = self.params

        updates = []
        gradients = {}
        for key in parameters.keys():
            gradient = T.grad(cost=cost, wrt=parameters[key])
            gradients[key] = gradient

        # Get the norm of the gradients
        norm = T.sqrt(T.sum(
            [T.sum(param_gradient ** 2) for param_gradient in gradients.values()]
            ))

        keys = []
        values = []
        for key in parameters.keys():
            gradients[key] = T.switch(
                    T.ge(norm, 1),             # 1 is the clipping value
                    gradients[key] / norm * 1,       # TODO: change it to a variable
                    gradients[key]
            )
            keys.append(key)
            values.append(gradients[key])

        return keys, values 


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
        self.downloadModel()

        # build test Model
        if (isinstance(testInput, theano.compile.sharedvalue.SharedVariable)):
            testInput = testInput.get_value()

        if self._testModel == None:
            self._testModel = theano.function(
                inputs=[self._x],
                outputs = self.y_pred,
            )

        # loop through the input and compute prediction
        preditction = self._testModel(testInput)
        return preditction 


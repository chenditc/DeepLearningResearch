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

    def __init__(self):
        self._pretrainModel = None

    @staticmethod
    def loadModelByName(name):
        module = __import__(name)
        className = getattr(module, name)
        # The dimension here should be dummy, does not influence the model
        classInstance = className(2,2)
        return classInstance


    ##
    # @brief                    Run the index to go through the training set  
    #
    # @return 
    def trainModel(self):

        # if the model is empty, build the trainModel
        if (self._trainModel == None):
            print "You have not build any training model yet."
            quit()

        # train the minibatchs
        for minibatch_index in xrange(self._totalBatches):
            minibatch_avg_cost = self._trainModel(minibatch_index)

    def pretrainModel(self):
        # if the model is empty, build the trainModel
        if self._pretrainModel == None:
            print "You have not build any pre-training model yet."
            return
        # train the minibatchs
        for minibatch_index in xrange(self._totalBatches):
            minibatch_avg_cost = self._pretrainModel(minibatch_index)

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

        if self._testModel == None:
            self._testModel = theano.function(
                inputs=[self._x],
                outputs=self.y_pred,
            )

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
            self.params[key].set_value(parameters[key])

    ##
    # @brief                Download the paramters from database and set them in shared variable 
    #
    # @param dataLoader
    #
    # @return 
    def downloadModel(self, dataLoader):
        cursor = dataLoader.getDatabaseCursor()
        cursor.execute('SELECT parameters FROM DeepLearningDB1.Model1 WHERE data_id = %s AND model_name = %s ORDER BY error_rate ASC LIMIT 1',
                        (dataLoader._data_id, self.__class__.__name__))
        dataRows = cursor.fetchall()
        jsonString = dataRows[0][0]
        self.loadModelFromJson(jsonString)


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
    # @param dataLoader     the loader will be used to upload data
    # @param errorRate      error rate of database
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

        print "Build training model to train:"
        print onlyTrain

        updates = []
        gradients = {}
        for key in onlyTrain:
            gradient = T.grad(cost=cost, wrt=parameters[key])
            gradients[key] = gradient

        # Get the norm of the gradients
        norm = T.sqrt(T.sum(
            [T.sum(param_gradient ** 2) for param_gradient in gradients.values()]
            ))
        clipped_gradients = {}
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



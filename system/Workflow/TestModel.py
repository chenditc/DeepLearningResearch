import sys
import DataLoader


class TestModel:
    ##
    # @brief            
    #
    # @param model          A instance of Learning Model
    # @param model_id       A id of learning model on database
    #
    # @return 
    def __init__(self, model = None, model_id = None):
        # Set the model
        self._model = model

        # TODO: if the model id is set, load the model from database
        if model_id != None:
            self._model = None 

    def testModel(self):
        while (1):
            line = sys.stdin.readline()
            if ',' not in line:
                print "Online test finish"
                break
            dataLoader = DataLoader.DataLoader()
            dataRow = dataLoader.parseRow(line)
            print "For data: ", dataRow
            print "Test result: ", self._model.testModel([dataRow])
            print "================================"


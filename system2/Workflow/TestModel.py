import sys
import DataLoader
import Model


class TestModel:
    ##
    # @brief            
    #
    # @param model          A instance of Learning Model
    # @param model_id       A id of learning model on database
    #
    # @return 
    def __init__(self, model = None, model_name = None, data_id = None):
        # Set the model
        self._model = model
        self._dataLoader = None

        if model_name != None and data_id != None:
            self.loadModel(model_name, data_id)

        if self._model == None:
            print "Initialize TestModel failed"
            quit()

    ##
    # @brief            Load model from database
    #
    # @param model_id   the model id and data id that used to select parameters
    # @param data_id
    #
    # @return 
    def loadModel(self, model_name, data_id):
        # initialize dataLoader
        if self._dataLoader == None:
            self._dataLoader = DataLoader.DataLoader(data_id)
        self._model = Model.Model.loadModelByName(model_name)

        # load parameters
        self._model.downloadModel(self._dataLoader)

    def testModel(self):
        while (1):
            line = raw_input('Please type input:\n')
            if ',' not in line:
                print "Online test finish"
                break
            dataRow = DataLoader.DataLoader.parseRow(line)
            print "Get data with size %d:" % len(dataRow), dataRow
            print "Test result: ", self._model.testModel([dataRow])
            print "================================"

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Testing Entrance.')
    parser.add_argument('--data_id', dest='data_id', help='the data used to train')
    parser.add_argument('--model_name', dest='model_name', help='the model used to train')
    args = parser.parse_args()

    if (args.data_id == None or args.model_name == None):
        parser.print_help()
        quit()

    testModel = TestModel(model_name = args.model_name, data_id = args.data_id) 
    testModel.testModel()


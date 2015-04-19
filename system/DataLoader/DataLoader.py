#!/usr/bin/python

# Data Loader Module that load data from database. The database configuration file is stored in config directory

import MySQLdb 
import json
import numpy
import os.path
import theano
import theano.tensor as T

class formatError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Incorrect format for row: " + json.dumps(self.value) )

class orderError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataLoader:

####################### Utility functions ######################

    def __init__(self, dataset, isClassifier = True, training_split = 0.6, validation_split = 0.2, test_split = 0.2):
        # private variables:
        self._dbCursor = None
        self._dbConnector = None
        self._dataMatrix = None
        self._split_n = None
        self._trainSetIndex = 0

        # TODO: automaticly figure out how much a batch should be
        self._dataBatch = 500
        self._training_split = training_split
        self. _validation_split = validation_split
        self._test_split = test_split
        self._data_id = dataset
        self._isClassifier = isClassifier

        # initialization functions
        self.peekMaximumRowID()
        self.peekDataDimension()
        if (training_split + validation_split + test_split != 1):
            raise test_split("The training_split + validation_split + test_split is not 1")

    ##
    # @brief    Lazy initialization for database cursor    
    #
    # @return 
    def getDatabaseCursor(self):	
        if ( None == self._dbCursor ):
            self._dbCursor = self.getDatabaseConnector().cursor() 
        return self._dbCursor

    ##
    # @brief    Lazy initialization for database connector
    #
    # @return 
    def getDatabaseConnector(self):
        if ( None == self._dbConnector ):
            # TODO: use configuration file to control user name and password
            self._dbConnector = MySQLdb.connect(host="deeplearningdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                                user="research", 
                                                passwd="Research013001",
                                                db="DeepLearningDB1")
        return self._dbConnector
    
    ##
    # @brief     Run all the query in que, 
    #            make sure they are in sync with database
    #
    # @return 
    def commitData(self):
        self.getDatabaseConnector().commit()
        return

    ##
    # @brief                Serialize the number array so it can be save into database
    #
    # @param numberArray    Input number array, should be the format of native float
    #
    # @return               a string represent the number 
    def encodeNumberArray(self, numberArray):
        return json.dumps(numberArray)


    ##
    # @brief                    de-serialize the number array
    #
    # @param numberArrayString  a number array in json format
    #
    # @return                   an array of float
    def decodeNumberArray(self, numberArrayString):
        return json.loads(numberArrayString)
   
    ##
    # @brief            Insert one row into training data table
    #
    # @param data_id    Id of data set where it belongs to, id is string
    # @param row_id     row_id define the sequence of the data
    # @param x          x value of training set, input of model
    # @param y          y value of training set, output of model
    #
    # @return 
    def insertOneRow(self, data_id, row_id, x, y):
        cursor = self.getDatabaseCursor()
        cursor.execute('INSERT INTO TrainingData1 ( data_id, row_id, x, y ) VALUES (%s, %s, %s, %s)',
                        (data_id, row_id, x, y))


    ##
    # @brief            Upload each row in dataMatrix as a training case. 
    #
    # @param dataMatrix An array of traning cases. 
    #                   The last element is the label. 
    #                   So each row must have at least 2 elements
    #
    # @param split_n    The last n number is the label, default is 1 
    #
    # @return 
    def uploadData(self, dataMatrix, data_id, split_n=1): 
        # split to each row
        row_id = 0
        for row in dataMatrix:
            if len(row) < 2:
                raise formatError(row)
            # split x and y, y is the last element
            x = row[:-split_n]
            y = row[-split_n:]
            # serialize x and y
            x = self.encodeNumberArray(x)
            y = self.encodeNumberArray(y)

            # insert one row into database
            self.insertOneRow(data_id, row_id, x, y)

            row_id += 1
        
        # sync database after all done
        self.commitData()
        return 


    ##
    # @brief            Fetch all data with data_id
    #
    # @param data_id    query key
    #
    # @return           a data matrix contains xy values and split_n 
    def downloadData(self, data_id):
        cursor = self.getDatabaseCursor()
        # query database:
        cursor.execute('SELECT row_id, x, y FROM TrainingData1 WHERE data_id = %s order by row_id asc', (data_id) )
        dataMatrix = []
        expect_row_id = 0
        split_n = 1
        
        # Error checking
        dataRows = cursor.fetchall()
        if len(dataRows) == 0:
            print "No data available for: ", data_id
            quit()

        for row in dataRows:
            row_id, x, y = row
            # check id
            if (expect_row_id != row_id):
                errorMessage = ( "Data Missing or Wrong.  expect:" 
                            + str(expect_row_id) + " ,get: " + str(row_id) )
                raise orderError(errorMessage)
            expect_row_id += 1

            x = self.decodeNumberArray(x)
            y = self.decodeNumberArray(y)
            # set split_n
            split_n = len(y)
            # set row
            dataMatrix.append( x + y )

        # store the data in class
        self._data_id = data_id;
        self._dataMatrix = dataMatrix
        self._split_n = split_n
        return dataMatrix, split_n



    ##
    # @brief                        Parse one row 
    #
    # @param inputString            
    # @param columnDeliminator      Deliminator for column
    # @param "
    #
    # @return 
    @staticmethod
    def parseRow(inputString, columnDeliminator = ","):
        elements = inputString.split(columnDeliminator)
        dataRow = []
        for element in elements:
            if ("" == element):
                continue
            dataRow.append(float(element))
        return dataRow

    ##
    # @brief                    Parse the data matrix from a file
    #
    # @param inputString        a file as string
    # @param rowDeliminator     The deliminator for each row, default is "\n"
    # @param columnDeliminator  The deliminator for each column, default is ","
    #
    # @return 
    def parseMatrix(self, inputString, rowDeliminator = "\n" , columnDeliminator = ","):
        stringRows = inputString.split(rowDeliminator)
        dataMatrix = []
        for stringRow in stringRows:
            if ("" == stringRow):
                continue
            dataRow = DataLoader.parseRow(stringRow, columnDeliminator)
            dataMatrix.append(dataRow)
        return dataMatrix

    ##
    # @brief            parse the file and upload the data using file name as data key
    #
    # @param filePath
    #
    # @return 
    def uploadDataFile(self, filePath):
        try:
            fileName = os.path.basename(filePath)
            fileString = open(filePath).read()
            dataMatrix = self.parseMatrix(fileString)
            self.uploadData(dataMatrix, data_id = fileName, split_n=1)
            return 0 
        except Exception as error:
            if 1062 == error[0] :
                print error
                print "The key is duplicated: ", fileName
                return 1
            else:
                print error
                raise error

    ##
    # @brief            Remove a set of data by data_id
    #
    # @param data_id    The id to identify which set of data to remove
    #
    # @return 
    def removeDataSet(self, data_id):
        cursor = self.getDatabaseCursor()
        cursor.execute('DELETE TrainingData1 FROM TrainingData1 WHERE data_id = %s', data_id)
        dataLoader.commitData();
        return 0 

############# Training related functions ##############################
 
    ##
    # @brief 
    #    Function that loads the dataset into shared variables
    #
    #    The reason we store our dataset in shared variables is to allow
    #    Theano to copy it into the GPU memory (when code is run on GPU).
    #    Since copying data into the GPU is slow, copying a minibatch everytime
    #    is needed (the default behaviour if the data is not in a shared
    #    variable) would lead to a large decrease in performance.
    #    
    # @param inputData      the input data to model 
    # @param outputData     the output data to model
    # @param borrow         if enable shallow copy or not
    # @param isClassifier   if this is a classifier
    #
    # @return 
    def shared_dataset(self, inputData, outputData, borrow=True, store=False):
        data_x = inputData
        data_y = outputData

        # Create share variable from numpy array
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        if (store == True):
            self._shared_x = shared_x
            self._shared_y = shared_y
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

   
    ##
    # @brief                Split the input data and output data
    #
    # @param dataMatrix     The matrix that contain rows of data
    # @param split_n        number of last columns as output
    #
    # @return 
    def splitInputAndOutput(self, dataMatrix, split_n):
        dataMatrix = numpy.array(dataMatrix)
        # Get first (len - split_n) column as input
        # The last column as training output
        inputColumn = range(len(dataMatrix[0]) - split_n ) 
        outputColumn = range(- split_n, 0)
        inputData = dataMatrix[:,inputColumn]
        outputData = dataMatrix[:,outputColumn]

        # if the data is for a classifier, use the first column of y only
        # and also change it to an array of scalar
        if self._isClassifier:
            outputData = outputData[:,0]

        return inputData, outputData


    ##
    # @brief            Get the maximum row id for a given data set
    #
    # @param data_id
    #
    # @return 
    def peekMaximumRowID(self):
        cursor = self.getDatabaseCursor()
        # query database:
        cursor.execute('SELECT max(row_id) as max_row_id FROM TrainingData1 WHERE data_id = %s', (self._data_id) )
        dataRows = cursor.fetchall()
        if len(dataRows) == 0:
            print "No data available for: ", data_id
            quit()
        # first row, first element
        self._maxRowID = dataRows[0][0]

    ##
    # @brief    check the input and output dimension
    #
    # @return 
    # TODO: add non-classifier dimension
    def peekDataDimension(self):
        cursor = self.getDatabaseCursor()
        # query database:
        cursor.execute('SELECT x, max(y) as max_y FROM TrainingData1 WHERE data_id = %s LIMIT 1', (self._data_id) )
        dataRows = cursor.fetchall()
        if len(dataRows) == 0:
            print "No data available for: ", data_id
            quit()
        # first row, first element
        x, y = dataRows[0]
        x = self.decodeNumberArray(x)
        y = self.decodeNumberArray(y)
        self._inDim = len(x)
        if self._isClassifier:
            # add one because the class number start from 1
            self._outDim = int(y[0]) + 1
        else:
            self._outDim = len(y)

    ##
    # @brief    Getter for input and output dimension
    #
    # @return   input dimension and output dimension
    def getDataDimension(self):
        return self._inDim, self._outDim

    ##
    # @brief            Given a range of start and end id, fetch the data set
    #
    # @param start      start row id
    # @param end        end row id
    #
    # @return 
    def downloadDataInRange(self, start, end):
        cursor = self.getDatabaseCursor()
        # query database:
        cursor.execute('SELECT row_id, x, y FROM TrainingData1 WHERE data_id = %s AND row_id >= %s AND row_id < %s order by row_id asc', (self._data_id, start, end) )
        dataMatrix = []
        expect_row_id = start
        split_n = 1
        
        # Error checking
        dataRows = cursor.fetchall()
        if len(dataRows) == 0:
            print "No data available for: ", data_id
            quit()

        for row in dataRows:
            row_id, x, y = row
            # check id
            if (expect_row_id != row_id):
                errorMessage = ( "Data Missing or Wrong.  expect:" 
                            + str(expect_row_id) + " ,get: " + str(row_id) )
                raise orderError(errorMessage)
            expect_row_id += 1

            x = self.decodeNumberArray(x)
            y = self.decodeNumberArray(y)
            split_n = len(y)
            # TODO: this seems to be able to release the reference of x and y,
            # But I don't understand why
            temp_result = numpy.array(x+y)
            dataMatrix.append(temp_result)

        return dataMatrix, split_n

    def updateTrainingSet(self):
        self._trainSetIndex = self._trainSetIndex % self._maxTrainBatchNumber
        start = self._trainSetIndex * self._dataBatch
        end = (self._trainSetIndex + 1) * self._dataBatch
         
        # get the data
        dataMatrix, split_n = self.downloadDataInRange(start, end)
        inputData, outputData = self.splitInputAndOutput(dataMatrix, split_n)

        self._shared_y.set_value(outputData)
        self._shared_x.set_value(inputData)

        self._trainSetIndex += 1

    ##
    # @brief    return training input and training output both as 2-D array
    #
    # @return 
    def getTrainingSet(self):
        # calculate index
        start = 0
        end = int(self._maxRowID * self._training_split)
        self._maxTrainBatchNumber = int( end / self._dataBatch)

        # get the data
        dataMatrix, split_n = self.downloadDataInRange(start, start + self._dataBatch)
        inputData, outputData = self.splitInputAndOutput(dataMatrix, split_n)
        return self.shared_dataset(inputData, outputData, store=True)

    ##
    # @brief    return validation input and training output both as 2-D array
    #
    # @return 
    def getValidationSet(self):
        start = int(self._maxRowID * self._training_split)
        end = int(self._maxRowID * ( self._training_split + self._validation_split))
        dataMatrix, split_n = self.downloadDataInRange(start, end)
        inputData, outputData = self.splitInputAndOutput(dataMatrix, split_n)
        return self.shared_dataset(inputData, outputData)

    ##
    # @brief    return validation input and training output both as 2-D array
    #
    # @return 
    def getTestSet(self):
        start = int(self._maxRowID * ( self._training_split + self._validation_split))
        end = self._maxRowID
        dataMatrix, split_n = self.downloadDataInRange(start, end)
        inputData, outputData = self.splitInputAndOutput(dataMatrix, split_n)
        return self.shared_dataset(inputData, outputData)




if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='data upload and download module.')
    parser.add_argument('-u', '--upload', dest='uploadFile', help='the data file to read and upload to database')
    parser.add_argument('-r', '--remove', dest='removeDataId', help='remove the data set with dataId')
    parser.add_argument('-t', '--test', dest='test_mode', action='store_true', help='the data file to read and upload to database')
    args = parser.parse_args()

    test_mode = args.test_mode
    uploadFile = args.uploadFile
    removeDataId = args.removeDataId

    if  len(sys.argv) == 1:
        print parser.print_help()
        quit()

    if (uploadFile != None):
        print "Start uploading", uploadFile
        dataLoader = DataLoader()
        rcode = dataLoader.uploadDataFile(uploadFile)
        if 0 == rcode:
            print "Upload success"
        

    if (removeDataId != None):
        dataLoader = DataLoader()
        rcode = dataLoader.removeDataSet(removeDataId)
        if 0 == rcode:
            print "Remove success"


    if (True == test_mode):
        print "===================="
        print "Test db connect"
        dataLoader = DataLoader()
        cursor = dataLoader.getDatabaseCursor()
        print "Connect success!"
        print "====================\n"

        # clean up
        dataLoader.removeDataSet("test_set")
        dataLoader.removeDataSet("test_sum_positive")
        print "===================="
        print "Test db select"
        cursor.execute('SELECT * FROM TrainingData1 WHERE data_id = "test_set"')
        print cursor.fetchall()
        print "SELECT result see above"
        print "====================\n"

        print "===================="
        print "Test db insert"
        cursor.execute('INSERT INTO TrainingData1 (data_id, row_id, x, y) VALUES ("test_set", 0, "x value", "y value")')
        cursor.execute('SELECT * FROM TrainingData1 WHERE data_id = "test_set"')
        dataLoader.commitData();
        print cursor.fetchall()
        print "Insert result see above"
        print "====================\n"

        print "===================="
        print "Test db delete"
        cursor.execute('DELETE TrainingData1 FROM TrainingData1 WHERE data_id = "test_set"')
        # print result
        cursor.execute('SELECT * FROM TrainingData1 WHERE data_id = "test_set"')
        dataLoader.commitData();
        print cursor.fetchall()
        print "Delete result see above"
        print "====================\n"

        print "===================="
        print "Test uploadData()"
        dataMatrix = [[1,2,3]]*10
        data_id = "test_set"
        dataLoader.uploadData(dataMatrix, data_id)
        # print result
        cursor.execute('SELECT * FROM TrainingData1 WHERE data_id = "test_set"')
        print cursor.fetchall()
        print "upload result see above"
        print "====================\n"

        print "===================="
        print "Test downloadData()"
        data_id = "test_set"
        dataMatrix, n = dataLoader.downloadData(data_id)
        print dataMatrix
        print n 
        print "download result see above"
        print "====================\n"

        print "===================="
        print "Test dataLoader.getTrainingSet()"
        data_id = "test_set"
        dataLoader.downloadData(data_id)
        print dataLoader.getTrainingSet()
        print "Result see above"
        print "====================\n"

        print "===================="
        print "Test dataLoader.getTrainingSet()"
        data_id = "test_set"
        dataLoader.downloadData(data_id)
        print dataLoader.getTrainingSet()
        print "Result see above"
        print "====================\n"

        print "===================="
        print "Test dataLoader.getValidationSet()"
        data_id = "test_set"
        dataLoader.downloadData(data_id)
        print dataLoader.getValidationSet()
        print "Result see above"
        print "====================\n"

        print "===================="
        print "Test dataLoader.getTestSet()"
        data_id = "test_set"
        dataLoader.downloadData(data_id)
        print dataLoader.getTestSet()
        print "Result see above"
        print "====================\n"

   

#!/usr/bin/python

# Data Loader Module that load data from database. The database configuration file is stored in config directory

import MySQLdb 
import json
import os.path

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

    def __init__(self):
        # private variables:
        self._dbCursor = None
        self._dbConnector = None

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
        if ( None == self._dbCursor ):
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
        for row in cursor.fetchall():
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
        return dataMatrix, split_n

    ##
    # @brief                        Parse one row 
    #
    # @param inputString            
    # @param columnDeliminator      Deliminator for column
    # @param "
    #
    # @return 
    def parseRow(self, inputString, columnDeliminator = ","):
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
            dataRow = self.parseRow(stringRow, columnDeliminator)
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='data upload and download module.')
    parser.add_argument('-u', '--upload', dest='uploadFile', help='the data file to read and upload to database')
    parser.add_argument('-r', '--remove', dest='removeDataId', help='remove the data set with dataId')
    parser.add_argument('-t', '--test', dest='test_mode', action='store_true', help='the data file to read and upload to database')
    args = parser.parse_args()

    test_mode = args.test_mode
    uploadFile = args.uploadFile
    removeDataId = args.removeDataId

if (uploadFile != None):
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
    dataMatrix = [[1,2,3],[4,5,6]]
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
    print "Test uploadDataFile()"
    dataLoader.uploadDataFile("test_sum_positive")
    print dataLoader.downloadData("test_sum_positive")[0][:10]
    print "10 parsed result see above"
    print "====================\n"

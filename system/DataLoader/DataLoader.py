#!/usr/bin/python

# Data Loader Module that load data from database. The database configuration file is stored in config directory

import MySQLdb 
import json

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
    # @return 
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

if __name__ == "__main__":

    print "===================="
    print "Test db connect"
    dataLoader = DataLoader()
    cursor = dataLoader.getDatabaseCursor()
    print "Connect success!"
    print "====================\n"

    # clean up
    cursor.execute('DELETE TrainingData1 FROM TrainingData1 WHERE data_id = "test_set"')
    dataLoader.commitData();
   
    print "===================="
    print "Test db select"
    cursor.execute("SELECT * FROM TrainingData1")
    print cursor.fetchall()
    print "SELECT result see above"
    print "====================\n"

    print "===================="
    print "Test db insert"
    cursor.execute('INSERT INTO TrainingData1 (data_id, row_id, x, y) VALUES ("test_set", 0, "x value", "y value")')
    cursor.execute("SELECT * FROM TrainingData1")
    dataLoader.commitData();
    print cursor.fetchall()
    print "Insert result see above"
    print "====================\n"

    print "===================="
    print "Test db delete"
    cursor.execute('DELETE TrainingData1 FROM TrainingData1 WHERE data_id = "test_set"')
    # print result
    cursor.execute("SELECT * FROM TrainingData1")
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
    cursor.execute("SELECT * FROM TrainingData1")
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

   



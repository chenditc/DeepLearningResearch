#!/usr/bin/python
import argparse
import sys
import json
import os

import MySQLdb
from boto import kinesis
import redis

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='data upload')
    parser.add_argument('-d', '--data', dest='data', help='the data file to read and upload to database')
    parser.add_argument('-b', '--batch', default = 20, dest='batch', help='the batch size of to insert')

    args = parser.parse_args()
    batchSize = int(args.batch)

    dbConnector = MySQLdb.connect(host="deeplearningdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                  user="research", 
                                  passwd="Research013001",
                                  db="DeepLearningDB1")
    kinesisConnection = kinesis.connect_to_region(region_name = 'us-west-2')


    dbCursor = dbConnector.cursor() 

    dbCursor.execute('SELECT row_id, x, y FROM TrainingData1 WHERE data_id = %s order by row_id asc', (args.data) )

    dataRows = dbCursor.fetchmany(size=batchSize)
    while len(dataRows) > 0:
        x = []
        y = []

        for dataRow in dataRows:
            x.append(json.loads(dataRow[1]))
            y += json.loads(dataRow[2])

        result = {'x':x,'y':y }
        result = json.dumps(result)
        x = []
        y = []

        print result
        kinesisConnection.put_record('Words5', result, '5')
        dataRows = dbCursor.fetchmany(size=batchSize)

         

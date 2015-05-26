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
    parser.add_argument('-b', '--batch', default = 2000, dest='batch', help='the batch size of to insert')

    args = parser.parse_args()


    redisClient = redis.StrictRedis(host='deeplearning-001.qha7wz.0001.usw2.cache.amazonaws.com', port=6379, db=0)
    dbConnector = MySQLdb.connect(host="deeplearningdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                  user="research", 
                                  passwd="Research013001",
                                  db="DeepLearningDB1")
    kinesisConnection = kinesis.connect_to_region(region_name = 'us-west-2')


    dbCursor = dbConnector.cursor() 

    dbCursor.execute('SELECT row_id, x, y FROM TrainingData1 WHERE data_id = %s order by row_id asc', (args.data) )

    dataRows = dbCursor.fetchall()
    x = []
    y = []

    for i in range(len(dataRows)):
        # Error checking
        
        dataRow = dataRows[i]
        x.append(json.loads(dataRow[1]))
        y += json.loads(dataRow[2])

        if sys.getsizeof(x) > 40000 or len(x) > args.batch:

            result = {'x':x,'y':y }
            result = json.dumps(result)
            x = []
            y = []

            print result
            kinesisConnection.put_record('Words5', result, '5')

     

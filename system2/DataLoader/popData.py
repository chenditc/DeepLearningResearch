#!/usr/bin/python
import argparse
import sys
import json
import os

import MySQLdb
from boto import kinesis

parser = argparse.ArgumentParser(description='data upload')
parser.add_argument('-d', '--data', dest='data', help='the data file to read and upload to database')
parser.add_argument('-r', '--row_id', dest='row_id', help='row id of data set')
args = parser.parse_args()


dbConnector = MySQLdb.connect(host="deeplearningdb1.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                              user="research", 
                              passwd="Research013001",
                              db="DeepLearningDB1")
dbCursor = dbConnector.cursor() 
dbCursor.execute('SELECT row_id, x, y FROM TrainingData1 WHERE data_id = %s and row_id = %s order by row_id asc', (args.data, args.row_id) )

# Error checking
dataRows = dbCursor.fetchall()[0]

x = json.loads(dataRows[1])
y = json.loads(dataRows[2])

result = {'x':[x],'y':y}
result = json.dumps(result)

print result

conn = kinesis.connect_to_region(region_name = 'us-west-2')
conn.put_record('Words5', result, '2')



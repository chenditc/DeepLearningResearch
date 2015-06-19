#!/usr/bin/python
#coding=utf-8

# import modules & set up logging
import gensim, logging
import re
import os
import sys
import argparse
import MySQLdb
import _mysql_exceptions
import json

def uploadVector(securityFile):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    dbConnector = MySQLdb.connect(host="stockdb.cafr6s1nfibs.us-west-2.rds.amazonaws.com", 
                                        user="chenditc", 
                                        passwd="cd013001",
                                        db="ch_day_tech")
    cursor = dbConnector.cursor()

    lines = open(securityFile).read().split('\n')
    Ticker = lines[0]
    Header = lines[1]
    dataLines = lines[2:]

    for dataLine in dataLines:
        if '#' in dataLine or len(dataLine) < 10:
            logging.info('Skipping line:' + dataLine)
            continue
        datapoints = tuple(re.split(r'\s', dataLine)[:-1])
        sql = 'INSERT INTO data (Ticker, Date, PX_OPEN, PX_LAST, PX_HIGH, PX_LOW, ATR, BB_MA, BB_UPPER, BB_LOWER, BB_WIDTH, BB_PERCENT, CMCI, DMI_PLUS, DMI_MINUS, ADX, ADXR, EMAVG, FEAR_GREED, HURST, MIN, MAX, MM_RETRACEMENT, MOMENTUM, MOM_MA, MACD, MACD_DIFF, MACD_SIGNAL, MAE_MIDDLE, MAE_UPPER, MAE_LOWER, MAOsc, MAO_SIGNAL, MAO_DIFF, PTPS, ROC, RSI, SMAVG, TAS_K, TAS_D, TAS_DS, TAS_DSS, TMAVG, VMAVG, WMAVG, WLPR) VALUES ("SHCOMP Index", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)' 
        try:
            cursor.execute(sql, datapoints)
            dbConnector.commit();
        except _mysql_exceptions.IntegrityError as e:
            if e[0] != 1062:
                logging.error(e)

            
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='Training word2vec.')
    parser.add_argument('-d', '--dir', dest='dir', help='directory that store the news files')

    args = parser.parse_args()

    if (args.dir == None):
        parser.print_help()
        quit()

    uploadVector(args.dir)


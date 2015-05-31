#!/usr/bin/python

import cPickle
import time
import multiprocessing

import storm
import redis
import numpy


def updateToRedis(updateQueue):
    redisClient = redis.StrictRedis(host='deeplearning-001.qha7wz.0001.usw2.cache.amazonaws.com', port=6379, db=0)
    while True:
        variableName, value = updateQueue.get()
        variableValueString = cPickle.dumps(value.tolist())
        redisClient.set(variableName, variableValueString)



class UpdateGradient(storm.BasicBolt):
    def initialize(self, stormconf, context):
        # initialize redis connection
        # try aws connection, if failed, use local connection
        # TODO: use more flexible config
        self.redisClient = None
        try:
            self.redisClient = redis.StrictRedis(host='deeplearning-001.qha7wz.0001.usw2.cache.amazonaws.com', port=6379, db=0)
            self.redisClient.ping()
        except:
            self.redisClient = redis.StrictRedis(host='127.0.0.1', port=6379, db=0) 

        self.lastValue = {}

        # TODO: initialize calculation queue

        # initialize updating quue
        self.updateQueue = multiprocessing.Queue() 
        updateProcess = multiprocessing.Process(target=updateToRedis, args=(self.updateQueue,))
        updateProcess.start()

    def process(self, tup):
        t1 = time.time()

        values = tup.values
        variableName = cPickle.loads(str(values[0]))
        gradient = cPickle.loads(str(values[1]))

        if variableName not in self.lastValue:
            # load matrix from redis
            oldValue = numpy.asarray(cPickle.loads(self.redisClient.get(variableName)))
        else:
            oldValue = self.lastValue[variableName]
        
        # update it
        newValue = oldValue - gradient * 0.01 

        self.lastValue[variableName] = newValue 

        # clean up que until only one value left
        self.updateQueue.put((variableName, newValue))
        if self.updateQueue.qsize() > 1:
            self.updateQueue.get()

        t2 = time.time()
        storm.log("Processing time in update bolt " + variableName + " : " + str(t2-t1))



UpdateGradient().run()

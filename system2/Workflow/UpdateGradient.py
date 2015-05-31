#!/usr/bin/python

import cPickle
import time

import storm
import redis
import numpy

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

    def process(self, tup):
        t1 = time.time()

        values = tup.values
        variableName = cPickle.loads(str(values[0]))
        storm.log("storm variable name:" + variableName)
        gradient = cPickle.loads(str(values[1]))

        if variableName not in self.lastValue:
            # load matrix from redis
            try:
                oldValue = numpy.asarray(cPickle.loads(self.redisClient.get(variableName)))
            except:
                oldValue = numpy.asarray(0)
        else:
            oldValue = self.lastValue[variableName]
        
        # update it
        newValue = oldValue - gradient * 0.01 

        self.lastValue[variableName] = newValue 

        self.redisClient.set(variableName, cPickle.dumps(newValue.tolist()))

        t2 = time.time()
        storm.log("Processing time in update bolt " + variableName + " : " + str(t2-t1))



UpdateGradient().run()

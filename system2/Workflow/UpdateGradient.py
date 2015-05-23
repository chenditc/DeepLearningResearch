#!/usr/bin/python

import json

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


    def process(self, tup):
        values = tup.values
        variableName = values[0]
        gradient = numpy.asarray(json.loads(values[1]))

        # load matrix from redis
        try:
            oldValue = numpy.asarray(json.loads(self.redisClient.get(variableName)))
        except:
            oldValue = numpy.asarray(0)
        
        # update it
        newValue = oldValue + gradient 

        storm.log(variableName + " new value:" + json.dumps(newValue.tolist()))

        self.redisClient.set(variableName, json.dumps(newValue.tolist()))

        storm.log(json.dumps(newValue.tolist()))


UpdateGradient().run()

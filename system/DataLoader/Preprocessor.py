#!/usr/bin/env python

import json

class Preprocessor():

    def __init__(self):
        return

    ##
    # @brief                Serialize the number array so it can be save into database
    #
    # @param numberArray    Input number array, should be the format of native float
    #
    # @return               a string represent the number 
    def encodeNumberArray(self, numberArray):
        # transform all number to int
        numberArray = [int(number) for number in numberArray]
        # transform all number to word
        wordArray = [self.indexToWord.get(str(number), 'UNKNOWN_WORD') for number in numberArray]
        return json.dumps(wordArray)


    ##
    # @brief                    de-serialize the number array
    #
    # @param numberArrayString  a number array in json format
    #
    # @return                   an array of float
    def decodeNumberArray(self, numberArrayString):
        wordArray = json.loads(numberArrayString)
        
        # transform all word to int
        numberArray = [self.wordToIndex.get(word, 0) for word in wordArray]

        # transform all int to float
        numberArray = [int(number) for number in numberArray]

        return numberArray


#!/usr/bin/env python

import json
import re

def preprocessText(text):
    # add space to make it easier to split in the future
    text = re.sub(r'([^a-z0-9A-Z ])([a-z0-9A-Z])', r'\1 \2', text) 
    text = re.sub(r'([a-z0-9A-Z])([^a-z0-9A-Z ])', r'\1 \2', text)
    return text

def createWordDictionary(textFile):
    wordToIndex = {}
    indexToWord = {}

    outputFile = textFile + "_indexToWordDict.txt"

    text = open(textFile).read()
    text = preprocessText(text)
    words = re.split('\ +', text)

    wordCount = 0
    for word in words:
        if word not in wordToIndex:
            wordToIndex[word] = wordCount
            indexToWord[wordCount] = word
            wordCount += 1

    output = open(outputFile, 'w')
    output.write(json.dumps(indexToWord))
    return wordToIndex, indexToWord

def splitTextIntoDataWindow(textFile, windowSize, wordToIndex, indexToWord):
    outputFile = textFile + "_data.txt"
    output = open(outputFile, 'w')

    dataSet = []
    text = open(textFile).read()
    text = preprocessText(text)
    words = re.split('\ +', text)

    # construct text windows, iterate and discard last (windowSize+1) words,
    for i in range(len(words) - windowSize):
        line = words[i : i+windowSize + 1]     # last one is the label
        line = [str(wordToIndex.get(word, '0')) for word in line]
        print >>output, ','.join(line)


class TextPreprocessor():

    def __init__(self, dictFile):
        self.indexToWord = json.loads(open(dictFile).read())
        self.wordToIndex = dict([(value, key) for key, value in self.indexToWord.items()])

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
        
        #TODO
        print wordArray

        # transform all word to int
        numberArray = [self.wordToIndex.get(word, 0) for word in wordArray]

        # transform all int to float
        numberArray = [int(number) for number in numberArray]

        return numberArray


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='text data manipulator')
    parser.add_argument('-f', '--file', dest='textFile', help='the text file to analyze and prepare dictionary')
    parser.add_argument('-d', '--dict', dest='dictFile', help='the dictionary file map from index to word')
    parser.add_argument('-w', '--windowSize', dest='windowSize', help='The window size of one set of text')

    args = parser.parse_args()

    if args.textFile != None:
        wordToIndex, indexToWord = createWordDictionary(args.textFile)    

        # if windows size specified, add split the data and put the data into a file
        if args.windowSize != None:
            splitTextIntoDataWindow(args.textFile, int(args.windowSize), wordToIndex, indexToWord)


    if args.dictFile != None:
        preprocessor = TextPreprocessor(args.dictFile)

        words = '["hello", "world"]'
        numbers = preprocessor.decodeNumberArray(words)
        newWords = preprocessor.encodeNumberArray(numbers)
        assert(words == newWords)
 

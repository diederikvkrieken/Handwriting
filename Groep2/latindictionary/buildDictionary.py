__author__ = 'diederik'

import sys
import pickle


class DictionaryBuilder:

    # Initializes the recognizer by initializing all parts of the pipeline
    def __init__(self):
        pass

    # This will write a dictionary file in the following format [fullword, [segmentedlabels]] so for example -> ["que", ['q', 'u', 'e']]
    def writeWordsDict(self, words, name):

        outputArray = []
        for word in words:

            segs = []
            for char in word[3]:
                segs.append(char[1])

            outputArray.append([word[1], segs])

        testArrayFile = open( name, 'w' )
        pickle.dump(outputArray, testArrayFile)
        testArrayFile.close()

    # This will write all the features to the dictionary in the following format [fullWord, [Feature 1 prediction], [feature 2 prediction], etc]
    # an example  ["que", ['q_', 'a', 'e'], ['q', 'u', 'i']]
    def writeFeatDict(self, predictions, words, name):

        outputArray = []

        for word in words:
            outputArray.append([word[1]])

        for prediction in predictions:
            countWord = 0
            for word in prediction:
                outputArray[countWord].append(word)
                countWord += 1

        testDictFile = open( name, 'w' )
        pickle.dump(outputArray, testDictFile)
        testDictFile.close()

    # This will return all the files
    def readDict(self, name):

        file = open(name, 'r')
        dict = pickle.load(file)
        file.close()

        return dict



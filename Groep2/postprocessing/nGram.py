__author__ = 'jasper'

from sklearn.feature_extraction.text import CountVectorizer
from Groep2.latindictionary import buildDictionary

import numpy as np

class Ngram():

    def __init__(self, document, n):
        if n < 2:
            assert False % "Ngram must be higher than 1"


        # Read the dictionary
        dictionInput = buildDictionary.DictionaryBuilder().readDict(document)

        # Reformat dictionary
        reformatedDict = []
        for word in dictionInput:
            tempString = ""
            for i in range(n - 1):
                tempString += "start "
            for char in word[1]:
                #print char[0]
                tempString += (char[:] + " ")

            tempString = tempString[:-1]
            reformatedDict.append(tempString)

        self.V = self.getVocabularySize(reformatedDict)
        self.nGramCountDictionary = self.getnGramCountDictionary(reformatedDict, n)
        self.ngramMinusOneCountDictionary = self.getnGramCountDictionary(reformatedDict, n -1)
        self.n = n

    def getVocabularySize(self, document):
        vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer= 'word', lowercase=False, token_pattern='(?u)[^ \\n]+')
        X = vectorizer.fit_transform(document)
        return len(vectorizer.get_feature_names())


    def getnGramCountDictionary(self, document, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), analyzer= 'word', lowercase=False, token_pattern='(?u)[^ \\n]+')
        X = vectorizer.fit_transform(document)
        terms = vectorizer.get_feature_names()
        freqs = X.sum(axis=0).A1
        return dict(zip(terms, freqs))


    def getProbability(self, sequence):
        if len(str(sequence).split()) != self.n:
            assert False % "sequence must have n parts"

        # plus one for smoothing
        if sequence in self.nGramCountDictionary:
            numerator = self.nGramCountDictionary[sequence] + 1
        else:
            numerator = 1
        # get the sequence without the last word:
        sequenceWithoutLastWord = str(sequence).rsplit(' ', 1)[0]
        #plus vocabulary size for smoothing
        if sequenceWithoutLastWord in self.ngramMinusOneCountDictionary:
            denominator = self.ngramMinusOneCountDictionary[sequenceWithoutLastWord] + self.V
        else:
            denominator = self.V
        P = numerator / float(denominator)

        return P

#document =  ['j o h n i s a g u y\np e r s o n a g u y']
document = "../KNMPDICT.dat"
ngram = Ngram(document, 3)
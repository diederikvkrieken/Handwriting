__author__ = 'jasper'

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time

import numpy as np

class Ngram():

    def __init__(self, document, n):
        if n < 2:
            assert False % "Ngram must be higher than 1"

        self.V = self.getVocabularySize(document)
        self.nGramCountDictionary = self.getnGramCountDictionary(document, n)
        self.ngramMinusOneCountDictionary = self.getnGramCountDictionary(document, n -1)
        self.n = n

    def getVocabularySize(self, document):
        vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern='(?u)\\b\\w+\\b')
        X = vectorizer.fit_transform(document)
        return len(vectorizer.get_feature_names())


    def getnGramCountDictionary(self, document, n):
        vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern='(?u)\\b\\w+\\b')
        X = vectorizer.fit_transform(document)
        terms = vectorizer.get_feature_names()
        freqs = X.sum(axis=0).A1
        return dict(zip(terms, freqs))


    def getProbability(self, sequence):
        if len(str(sequence).split()) != self.n:
            assert False % "sequence must have n parts"

        # plus one for smoothing
        numerator = self.nGramCountDictionary[sequence] + 1
        # get the sequence without the last word:
        sequenceWithoutLastWord = str(sequence).rsplit(' ', 1)[0]
        #plus vocabulary size for smoothing
        denominator = self.ngramMinusOneCountDictionary[sequenceWithoutLastWord] + self.V
        P = numerator / float(denominator)

        return P

document =  ['john is a guy', 'person a guy']
ngram = Ngram(document, 3)
P = ngram.getProbability("john is a")
print P

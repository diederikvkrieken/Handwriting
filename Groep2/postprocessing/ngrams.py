__author__ = 'diederik'

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from latindictionary import buildDictionary


class Ngrams():

    def __init__(self):
        pass

    def buildNgram(self, fileName):

        # Read the dictionary
        dictionInput = buildDictionary.DictionaryBuilder().readDict(fileName)

        # Reformat dictionary
        reformatedDict = []
        for word in dictionInput:
            tempString = ""
            for char in word[1]:
                 #print char[0]
                tempString += (char[:] + " ")

            tempString = tempString[:-1]
            reformatedDict.append(tempString)

        # initiate ngram_vectorizer
        ngram_vectorizer = CountVectorizer(analyzer= 'word', ngram_range=(2, 2), token_pattern='\\b\\w+\\b')

        # Fit.
        counts = ngram_vectorizer.fit_transform(reformatedDict)

        # print ngram_vectorizer.get_feature_names()
        #print counts.toarray().astype(int)

        # Get the differnt "grams"
        matrix_terms = np.array(ngram_vectorizer.get_feature_names())

        # Use the axis keyword to sum over rows
        matrix_freq = np.asarray(counts.sum(axis=0)).ravel()

        # temp = (float(len(matrix_freq))
        dictNgram = dict(zip(matrix_terms, (matrix_freq / float(sum(matrix_freq)))))

        return dictNgram
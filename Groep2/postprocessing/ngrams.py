__author__ = 'diederik'

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from latindictionary import buildDictionary


class Ngrams():

    def __init__(self):
        pass

    # This will build an N-Gram given a .dat file For example KNMPDICT.dat. Type will be either 2 or 3 for digram and trigram.
    def buildNgram(self, fileName, type):

        # Read the dictionary
        dictionInput = buildDictionary.DictionaryBuilder().readDict(fileName)

        # Reformat dictionary
        reformatedDict = []

        # initiate ngram_vectorizer
        if type == 2:
            ngram_vectorizer = self.diNgram()

            for word in dictionInput:
                tempString = "start "
                for char in word[1]:
                    #print char[0]
                    tempString += (char[:] + " ")

                tempString = tempString[:-1]
                print tempString
                reformatedDict.append(tempString)

        elif type == 3:
            ngram_vectorizer = self.triNgram()

            for word in dictionInput:
                tempString = "start start "
                for char in word[1]:
                    #print char[0]
                    tempString += (char[:] + " ")

                tempString = tempString[:-1]
                reformatedDict.append(tempString)
        else:
            assert False % "Ngram Type must either be 2 or 3"

        # Fit.
        counts = ngram_vectorizer.fit_transform(reformatedDict)

        # Get the differnt "grams"
        matrix_terms = np.array(ngram_vectorizer.get_feature_names())

        # Use the axis keyword to sum over rows
        matrix_freq = np.asarray(counts.sum(axis=0)).ravel()

        # temp = (float(len(matrix_freq))
        dictNgram = dict(zip(matrix_terms, (matrix_freq / float(sum(matrix_freq)))))

        return dictNgram

    def diNgram(self):
        return CountVectorizer(analyzer= 'word', ngram_range=(2, 2), lowercase=False, token_pattern='(?u)[^ \\n]+')

    def triNgram(self):
        return CountVectorizer(analyzer= 'word', ngram_range=(3, 3), lowercase=False, token_pattern='(?u)[^ \\n]+')

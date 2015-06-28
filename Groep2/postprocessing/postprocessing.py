__author__ = 'diederik'

import nGram
import ngramPredictions

class Postprocessing():

    def __init__(self):
        pass

    def ngramPostProcessing(self, segmentsOptions, document):

        ngram = nGram.Ngram(document, 3)
        return ngramPredictions.pathBuilder().run(segmentsOptions, ngram)

    def run(self, segmentsOptions, document = "../KNMPDICT.dat"):

        return self.ngramPostProcessing(segmentsOptions, document)





__author__ = 'diederik'

import nGram
import ngramPredictions

class Postprocessing():

    def __init__(self):
        pass

    # This will perform the ngram part.
    def ngramPostProcessing(self, segmentsOptions, document):

        ngram = nGram.Ngram(document, 3)
        return ngramPredictions.pathBuilder().run(segmentsOptions, ngram)

    # This function will reformat the predictions.
    def reformatSegmentsOptions(self, segmentOptions):

        rfsegmentsOptions = []

        for char in segmentOptions[0][0]:
            rfsegmentsOptions.append([])

        for segment in segmentOptions:
            charCount = 0
            for charOption in segment:
                rfsegmentsOptions[charCount].append(charOption)
                charCount = +1

        return rfsegmentsOptions

    def run(self, predictions, document = "../KNMPDICT.dat"):

        # Reformated the predictions.
        segmentsOptions = self.reformatSegementsOptions(predictions[0])

        # Return the results from the ngram matching.
        return self.ngramPostProcessing(segmentsOptions, document)





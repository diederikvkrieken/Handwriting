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

        """
        print "-----Segments----"
        for segments in segmentOptions:
            print segments
        """

        rfsegmentsOptions = []

        for words in segmentOptions:
            rfsegmentsOptions.append([])
            for char in words[0]:
                rfsegmentsOptions[-1].append([])

        wordCount = 0
        for words in segmentOptions:

            for segments in words:
                charCount = 0
                for charOption in segments:
                    # print "CHAR ", charCount, ": ", charOption
                    rfsegmentsOptions[wordCount][charCount].append(charOption)
                    charCount += 1

            wordCount += 1

        """
        print "------RF-------"
        wordCount = 0
        for words in rfsegmentsOptions:
            for options in words:
                print "WORD ", wordCount, ": ", options
            wordCount += 1
        print "---done---"
        """

        return rfsegmentsOptions

    def run(self, predictions, document = "KNMPDICT.dat"):

        # Reformated the predictions.
        segmentsOptions = self.reformatSegmentsOptions(predictions[0])

        # Return the results from the ngram matching.
        return self.ngramPostProcessing(segmentsOptions, document)





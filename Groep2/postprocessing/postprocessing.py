__author__ = 'diederik'

import nGram
import ngramPredictions

class Postprocessing():

    def __init__(self):
        pass

    # This will perform the ngram part.
    def ngramPostProcessing(self, segmentsOptions, document, type = 3):

        if type == 3:
            ngram = nGram.Ngram(document, 3)
            return ngramPredictions.pathBuilder().run(segmentsOptions, ngram)
        else:
            ngram = nGram.Ngram(document, 2)
            return ngramPredictions.pathBuilder().runDi(segmentsOptions, ngram)

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

    def getSingleChar(self, predictions):

        singleCharArray = [[],[]]
        popList = []

        wordCount = 0
        for word in predictions[0]:

            single = True

            if len(word) > 1:
                for char in word[0][1:]:
                    if char != '_':
                        single = False

            if single == True:

                singleWord = []

                for char in word[0]:
                    singleWord.append(char)

                singleCharArray[0].append(singleWord)
                popList.append(wordCount)

            wordCount += 1

        # Pop all single chars
        for index in popList:
            singleCharArray[1].append((predictions[1][index]))

        for offset, index in enumerate(popList):
            index -= offset
            predictions[0].pop(index)
            predictions[1].pop(index)

        singleCharWinner = []
        for word in singleCharArray:
            winner = ''
            for char in word[0][0]:
                if char == '_':
                    break
                winner = winner + char
            singleCharWinner.append(winner)

        return predictions, singleCharArray, singleCharWinner

    # This is the validate version.
    def getSingleCharValidate(self, predictions):

        singleCharArray = []
        popList = []

        wordCount = 0
        for word in predictions:

            single = True

            if len(word) > 1:
                for char in word[0][1:]:
                    if char != '_':
                        single = False

            if single == True:

                singleWord = []

                for char in word[0]:
                    singleWord.append(char)

                singleCharArray.append(singleWord)
                popList.append(wordCount)

            wordCount += 1

        for offset, index in enumerate(popList):
            index -= offset
            predictions.pop(index)

        singleCharWinner = []
        for word in singleCharArray:
            winner = ''
            for char in word[0][0]:
                if char == '_':
                    break
                winner = winner + char
            singleCharWinner.append(winner)

        return predictions, singleCharArray, singleCharWinner

    def run(self, predictions, document = "KNMPSTANFORDDICT.dat"):

        # Create a list with only the single chars
        predictions, singleChars, singleCharWinner = self.getSingleChar(predictions)

        # Reformated the predictions.
        segmentsOptions = self.reformatSegmentsOptions(predictions[0])

        # Return the results from the ngram matching.
        return self.ngramPostProcessing(segmentsOptions, document, type=3), singleChars, singleCharWinner

    def runValidate(self, predictions, document = "KNMPSTANFORDDICT.dat"):

        # Create a list with only the single chars
        predictions, singleChars, singleCharWinner = self.getSingleCharValidate(predictions)

        # Reformated the predictions.
        segmentsOptions = self.reformatSegmentsOptions(predictions)

        # Return the results from the ngram matching.
        return self.ngramPostProcessing(segmentsOptions, document, type=3), singleChars, singleCharWinner










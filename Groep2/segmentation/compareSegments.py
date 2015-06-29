__author__ = 'diederik'

# wordXML contains the coordinates of every letter for every word so: wordXml = [[left, right, label]] for every LETTER
class Comparator:

    def __init__(self):
        self.letterChange = True
        self.lastLetter = ""
        self.leftOutLeft = False
        pass

    def compare(self, csc_columns, wordXml, threshold=8):

        self.letterChange = True
        self.lastLetter = ""
        self.leftOutLeft = False
        self.letterAdded = False

        charCount = 0
        cscCount = 0

        csc_columnsWithLabel = []

        # Add the first csc
        csc_columnsWithLabel.append([csc_columns[0], ""])

        while True:

            if (len(wordXml)-1) < charCount:
                break

            threshold = int((wordXml[charCount][1] - wordXml[charCount][0]) / 3)

            # Situation 1
            if wordXml[charCount][1] <= csc_columns[cscCount]:

                if cscCount == 0 or wordXml[charCount][1] - csc_columns[cscCount-1] > threshold:
                    csc_columnsWithLabel[-1][1] = self.sameLetterCheck(cscCount, csc_columnsWithLabel, wordXml[charCount])
                    # print "Added: ", csc_columnsWithLabel[-1][1]

                    self.leftOutLeft = False
                    self.letterAdded = True
                else:
                    # Since we leave out the left char we do not want to also leave out the right char. This would result in an exmpty segment
                    self.leftOutLeft = True

                # If the char is not yet in the seg so not added at all than just still add it.
                if self.letterAdded == False:
                    csc_columnsWithLabel[-1][1] = self.sameLetterCheck(cscCount, csc_columnsWithLabel, wordXml[charCount])
                    # print "Added: ", csc_columnsWithLabel[-1][1]

                # Go to next char.
                self.letterChange = True
                charCount += 1
                self.letterAdded = False

                continue

            # Situation 2
            if wordXml[charCount][1] > csc_columns[cscCount]:

                if self.leftOutLeft == True or csc_columns[cscCount] - wordXml[charCount][0] > threshold:
                    csc_columnsWithLabel[-1][1] = self.sameLetterCheck(cscCount, csc_columnsWithLabel, wordXml[charCount])
                    # print "Added: ", csc_columnsWithLabel[-1][1]
                    self.leftOutLeft = False
                    self.letterAdded = True
                    # print "ADDED WORD IN SIT 2: ", wordXml[charCount]

                    # Break when we are out of segments
                    if (len(csc_columns)-1 == cscCount):
                        charCount += 1
                        # print "SEG BREAK IN IF!!!"
                        break

                # If the cscCount is still empty just add the last letter.
                if csc_columnsWithLabel[cscCount][1] == '':
                    csc_columnsWithLabel[-1][1] = self.sameLetterCheck(cscCount, csc_columnsWithLabel, wordXml[charCount])
                    # print "Added: ", csc_columnsWithLabel[-1][1]
                    self.letterAdded = True

                # Break when we are out of segments
                if (len(csc_columns)-1 == cscCount):
                    # print "SEG BREAK!"
                    break

                cscCount += 1
                csc_columnsWithLabel.append([csc_columns[cscCount], ""])

                continue


        cscString = ""
        for cscChar in csc_columnsWithLabel:
            cscString += cscChar[1]

        # print "Cscstring before: ", cscString

        # Add the last letters.
        while len(wordXml)-1 >= charCount:
            # print "Added as last: ", wordXml[charCount]
            csc_columnsWithLabel[-1][1] = self.sameLetterCheck(cscCount, csc_columnsWithLabel, wordXml[charCount])
            # print "Added: ", csc_columnsWithLabel[-1][1]
            charCount += 1
            # print "AFTER ADDED: ", wordXml[charCount]

        # If we have a last segment that is empty. I.e. due to thresholding
        if csc_columnsWithLabel[-1][1] == '':
            csc_columnsWithLabel[-1][1] = self.sameLetterCheck(cscCount, csc_columnsWithLabel, wordXml[-1])
            # print "Added: ", csc_columnsWithLabel[-1][1]

        # IF we have a wrong segmentation add the missed segments with garbage symbol.
        if len(csc_columnsWithLabel) != len(csc_columns):
            start = len(csc_columnsWithLabel)
            for csc in csc_columns[start:]:
                csc_columnsWithLabel.append([csc, '**GARBAGE**'])

        #--------------------------ASSERTION AND DEBUGGING CODE----------------------------------------
        for cscChar in csc_columnsWithLabel:
            assert cscChar[1] != '', "Somehow we got an empty segment this means there is a badly annotated word!"

        xmlString = ""
        for charXml in wordXml:
            xmlString += charXml[2]

        # print "XMLSTRING: ", xmlString
        cscString = ""
        for cscChar in csc_columnsWithLabel:
            cscString += cscChar[1]

        for charXml in xmlString:
            if charXml not in cscString:
                print "We ar missing characters in the CSC this is a badly annotated word!%s, %s" % (charXml, cscString)

        """
        print "-----COMP-----"
        print xmlString
        print cscString
        print '"------------'
        """

        return csc_columnsWithLabel



    # Will check whether segments are still the same letter
    def sameLetterCheck(self, n , csc_columnsWithLabel, charSeg):

        if n!= 0 and charSeg[2] in csc_columnsWithLabel[-2][1]:

            if self.letterChange == False:
                self.lastLetter = self.lastLetter + '_'

                return csc_columnsWithLabel[-1][1] + self.lastLetter
            else:
                self.letterChange = False
                self.lastLetter = charSeg[2]

                return csc_columnsWithLabel[-1][1] + charSeg[2]

        else:
            self.letterChange = False
            self.lastLetter = charSeg[2]

            return csc_columnsWithLabel[-1][1] + charSeg[2]




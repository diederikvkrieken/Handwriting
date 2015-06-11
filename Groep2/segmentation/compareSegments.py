__author__ = 'diederik'

# wordXML contains the coordinates of every letter for every word so: wordXml = [[top, bottom, left, right, label]] for every LETTER
class Comparator:

    def __init__(self):
        self.letterChange = True
        pass

    def compare(self, csc_columns, wordXml):

        csc_columnsWithLabel = []
        n = 0
        k = 0
        for charSeg in wordXml:

            for csc in csc_columns[n:]:

                #Only add a new csc when we do not have undersegmenting
                if len(csc_columnsWithLabel) == 0 or csc_columnsWithLabel[-1][0] != csc:
                    csc_columnsWithLabel.append([csc, ""])

                #If a csc has a smaller x value then the segments of the letter.
                if csc <= charSeg[1]:

                    csc_columnsWithLabel[-1][1] = self.sameLetterCheck(n, csc_columnsWithLabel, charSeg, k, wordXml)
                    n += 1

                #Else calculate how much percentage we overlap
                else:

                    if k != len(wordXml)-1:
                        diff = csc - charSeg[1]
                        cscOffset = csc - csc_columns[n-1]
                        if cscOffset == 0:
                            # print "something terriblle has happened!!<<<--------- See line 35 compareSegments.py FIX THIS!!"
                            # print "charSeg: ", charSeg[0], " x: ", charSeg[1], " SC x: ", csc ,"SC-1 x: ", csc_columns[n-1]
                            # print "SEGMENTS:", csc_columns, "WORD SEGMENTS", wordXml
                            csc_columnsWithLabel[-1][1] = self.sameLetterCheck(n, csc_columnsWithLabel, charSeg, k ,wordXml)
                            break

                        percentage = float(diff) / float(cscOffset)

                        if percentage < 0.5:

                            csc_columnsWithLabel[-1][1] = self.sameLetterCheck(n, csc_columnsWithLabel, charSeg, k ,wordXml)

                            # When we have undersegmenting break.
                            if len(wordXml) != (k+1) and csc >= wordXml[k+1][1]:
                                break

                            n += 1

                        break
                    else:
                        csc_columnsWithLabel[-1][1] = self.sameLetterCheck(n, csc_columnsWithLabel, charSeg, k ,wordXml)


            k += 1
            self.letterChange = True


        return csc_columnsWithLabel


    # Will check whether segments are still the same letter
    def sameLetterCheck(self, n , csc_columnsWithLabel, charSeg, k, wordXml):

        if n!= 0 and charSeg[2] in csc_columnsWithLabel[-2][1]:

            if self.letterChange == False:
                return csc_columnsWithLabel[-2][1] + '_'
            else:
                self.letterChange = False
                return csc_columnsWithLabel[-1][1] + charSeg[2]

        else:
            self.letterChange = False
            return csc_columnsWithLabel[-1][1] + charSeg[2]




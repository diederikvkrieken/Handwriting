__author__ = 'diederik'

from toolbox import wordio

# wordXML contains the coordinates of every letter for every word so: wordXml = [[top, bottom, left, right, label]] for every LETTER
class Comparator:

    def __init__(self):
        pass

    def compare(self, csc_columns, wordXml):

        csc_columnsWithLabel = []
        n = 0
        k = 0
        for charSeg in wordXml:

            for csc in csc_columns[n:]:

                #Only add a new csc when we do not have undersegmenting
                if n == 0 or csc_columnsWithLabel[-1][0] != csc:
                    csc_columnsWithLabel.append([csc, ""])

                #If a csc has a smaller x value then the segments of the letter.
                if csc <= charSeg[3]:

                    csc_columnsWithLabel[-1][1] = self.sameLetterCheck(n, csc_columnsWithLabel, charSeg, k, wordXml)
                    n += 1

                #Else calculate how much percentage we overlap
                else:
                    diff = csc - charSeg[3]
                    cscOffset = csc - csc_columns[n-1]
                    percentage = diff / cscOffset

                    if percentage < 0.5:

                        csc_columnsWithLabel[-1][1] = self.sameLetterCheck(n, csc_columnsWithLabel, charSeg, k ,wordXml)

                        # When we have undersegmenting break.
                        if len(wordXml) != (k+1) and csc >= wordXml[k+1][3]:
                            break
                        n += 1

                    break

            k += 1

        return csc_columnsWithLabel


    # Will check whether segments are still the same letter
    def sameLetterCheck(self, n , csc_columnsWithLabel, charSeg, k, wordXml):

        if n!= 0 and charSeg[4] in csc_columnsWithLabel[-2][1] and csc_columnsWithLabel[-2][0] <= wordXml[k-1][3] :
            return csc_columnsWithLabel[-2][1] + '*'
        else:
            return csc_columnsWithLabel[-1][1] + charSeg[4]


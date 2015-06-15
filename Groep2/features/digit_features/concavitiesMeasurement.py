__author__ = 'diederik'

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ConcavitiesMeasurement():

    def __init__(self):
        self.xSize = 16
        self.ySize = 18
        pass

    def scaleAndSegment(self, img):

        # Scale to 18 * 15 !MOOI!
        img = cv2.resize(img, (self.xSize, self.ySize), interpolation=cv2.INTER_NEAREST)

        # Segment
        segments = [[0,8,0,6],[0,8,6,12],[0,8,12,18],[8,16,0,6],[8,16,6,12],[8,15,12,18]]

        return img, segments

    def findConcavities(self, img, segments):

        # Initiate the segment features
        features = []

        #Calculate feature for every segment
        for seg in segments:
            features.append(self.calculateSegmentFeature(img,seg))

        #Combine feature
        featureMerged = []
        for f in features:
            featureMerged += f.tolist()

        return featureMerged

    def calculateSegmentFeature(self,img,seg):

        SegmentFeature = np.zeros(13)

        # iterate through every white pixel in segment
        for x in range(seg[0],seg[1]):
            for y in range(seg[2], seg[3]):
                if img[y,x] == 0:

                     # Calculate in what direction we find black pixels.
                    fDir = self.freemanPixel(img,x,y)

                    # Classify
                    if fDir[0] == 1 and fDir[1] == 1 and fDir[2] == 1 and fDir[3] == 1:
                        s = self.auxilaryDir(img,x,y)

                        if s[0] == True:
                            SegmentFeature[8] += 1
                        else:
                            SegmentFeature[9] += s[1]
                            SegmentFeature[10] += s[2]
                            SegmentFeature[11] += s[3]
                            SegmentFeature[12] += s[4]

                    elif fDir[0] == 1 and fDir[1] == 1 and fDir[2] == 1:
                        SegmentFeature[7] += 1
                    elif fDir[0] == 1 and fDir[1] == 1 and fDir[3] == 1:
                        SegmentFeature[6] += 1
                    elif fDir[0] == 1 and fDir[2] == 1 and fDir[3] == 1:
                        SegmentFeature[5] += 1
                    elif fDir[1] == 1 and fDir[2] == 1 and fDir[3] == 1:
                        SegmentFeature[4] += 1
                    elif fDir[1] == 1 and fDir[2] == 1:
                        SegmentFeature[3] += 1
                    elif fDir[0] == 1 and fDir[1] == 1:
                        SegmentFeature[2] += 1
                    elif fDir[0] == 1 and fDir[3] == 1:
                        SegmentFeature[1] += 1
                    elif fDir[2] == 1 and fDir[3] == 1:
                        SegmentFeature[0] += 1

        return SegmentFeature

    def freemanPixel(self, img, x, y):

        freemanDirections = [0,0,0,0]

        #direction 0
        if 1 in img[0:y,x]:
            freemanDirections[0] = 1

        # direction 1
        if 1 in img[y,x:self.xSize]:
            freemanDirections[1] = 1

        # direction 2
        if 1 in img[y:self.ySize,x]:
            freemanDirections[2] = 1

        # direction 4
        if 1 in img[y,0:x]:
            freemanDirections[3] = 1

        return freemanDirections

    def auxilaryDir(self, img, x, y):

        sList = [False, 1,1,1,1]

        #S1 direction
        x2 = x
        y2 = y

        while x2 != 0 and y2 != 0:
            x2 -= 1
            y2 -= 1

            if img[y2,x2] == 1:
                sList[1] = 0
                break

        #S2 direction
        x2 = x
        y2 = y

        while x2 != self.xSize-1 and y2 != 0:
            x2 += 1
            y2 -= 1

            if img[y2,x2] == 1:
                sList[2] = 0
                break

        #S3 direction
        x2 = x
        y2 = y

        while x2 != 0 and y2 != self.ySize-1:
            x2 -= 1
            y2 += 1

            if img[y2,x2] == 1:
                sList[3] = 0
                break

        #S4 direction
        x2 = x
        y2 = y

        while x2 != self.xSize-1 and y2 != self.ySize-1:
            x2 += 1
            y2 += 1

            if img[y2,x2] == 1:
                sList[4] = 0
                break

        #Check if we are in a concavity
        if sList[1] == 0 and sList[2] == 0 and sList[3] == 0 and sList[4] == 0:
            sList[0] = True

        return sList


    def run(self, img):
        print "Running Concavities Measurement"
        # preprocess
        img, segments = self.scaleAndSegment(img)

        # calculate feature
        feature = self.findConcavities(img, segments)

        # Testing purposes
        # fig, ax = plt.subplots()
        # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        # plt.show()
        # cv2.waitKey(0)
        return feature

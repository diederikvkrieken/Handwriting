import cv2
import numpy
import numpy as np
import math
import scipy
from bisect import bisect_left

class MAT_Grad():

    def __init__(self):
        self.xSize = 32
        self.ySize = 32
        self.orientations = [-180, -135,-90,-45,0,45,90,135,180]
        self.directions = [-135,-90,-45,1,45,90,135,180]
        pass

    def scale(self, img):
        # Scale to Sizes in Greyscale
        img = cv2.resize(img, (self.xSize, self.ySize))
        return img

    def closestValue(self, value):
        pos = bisect_left(self.orientations, value)
        if pos == 0:
            return self.orientations[0]
        if pos == len(self.orientations):
            return self.orientations[-1]
        before = self.orientations[pos - 1]
        after = self.orientations[pos]
        if after - value < value - before:
            return after
        else:
            return before

    def findMATGrad(self,img):
        sobelDirec = numpy.zeros((img.shape[0],img.shape[1],1), numpy.float)                                  #empty image

        #grab both edges for sobelx
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)

        #both edges for sobely
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

        for x in range(32):
            for y in range(32):
                px = sobelx[x][y]
                py = sobely[x][y]
                if math.sqrt((px*px)+(py*py))!=0:
                    sobelDirec[x][y] = math.atan2(py,px)*(180/math.pi)
                    pos = self.closestValue(sobelDirec[x][y])
                    if pos ==-180:
                        pos = 180
                    if pos == 0:
                        pos = 1
                    sobelDirec[x][y] = pos

        feature_vector = []
        for x in range(0,4):
            x = x*(32/4)
            for y in range(0,4):
                y = y*(32/4)
                a = sobelDirec[x:x + 8, y:y + 8]
                for p in self.directions:
                    count = (a==p).sum()
                    feature_vector.append(count)
        return feature_vector


    def run(self,image):
        # print "Running MAT Gradient"
        image = self.scale(image)
        feature = self.findMATGrad(image)
        #featureFirstOnly = [row[0] for row in feature]
        return feature

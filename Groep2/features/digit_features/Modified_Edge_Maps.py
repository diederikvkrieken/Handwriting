import cv2
import numpy
import numpy as np
from preprocessing import thinning, prepImage
from bisect import bisect_left
import math

class EdgeMaps():
    def __init__(self):
        self.xSize = 25
        self.ySize = 25
        self.orientations = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        self.directions = [45, 90, 135, 180]
        pass

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

    def findEdgeMaps(self,img):
        # load an color image in grayscale
        img = cv2.resize(img, (self.xSize,self.ySize))

        #the preprocessor object
        thin = thinning.thinning(img)

         #grab both edges for sobelx
        sobelx = cv2.Sobel(thin,cv2.CV_64F,1,0,ksize=-1)
        #abs_sobel64f = np.absolute(sobelx64f)
        #sobelx = np.uint8(abs_sobel64f)


        #both edges for sobely
        sobely = cv2.Sobel(thin,cv2.CV_64F,0,1,ksize=-1)
        #abs_sobel64f = np.absolute(sobely64f)
        #sobely = np.uint8(abs_sobel64f)

        sobelDirec = numpy.zeros((thin.shape[0],thin.shape[1],1), numpy.float)                                         #empty image

        width = thin.shape[1]
        height = thin.shape[0]

        for x in range(height):
            for y in range(width):
                px = sobelx[x][y]
                py = sobely[x][y]
                if math.sqrt((px*px)+(py*py))!=0:
                    sobelDirec[x][y] = math.atan2(py,px)*(180/math.pi)
                    pos = self.closestValue(sobelDirec[x][y])
                    if pos == 0 | pos == -180:
                        pos = 180
                    elif pos == -45:
                        pos = 135
                    elif pos == -90:
                        pos = 90
                    elif pos == -135:
                        pos = 45
                    sobelDirec[x][y] = pos

        feature_vector = []
        for x in range(0,5):
            x = x*5
            for y in range(0,5):
                y = y*5
                a = sobelDirec[x:x + 5, y:y + 5]
                for p in self.directions:
                    count = (a==p).sum()
                    feature_vector.append(count)
                original = thin[x:x + 5, y:y + 5]
                count = (original==1).sum()
                feature_vector.append(count)
        return feature_vector

    def run(self,img):
        print "Running Modified Edge Maps"
        feature = self.findEdgeMaps(img)
        return feature



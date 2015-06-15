import cv2
import numpy
import Image
import math
import scipy

class MAT_Grad():

    def __init__(self):
        self.xSize = 32
        self.ySize = 32
        pass

    def scale(self, img):
        # Scale to Sizes in Greyscale
        img = cv2.resize(img, (self.xSize, self.ySize))
        return img

    def findMATGrad(self,img):
        # load an color image in grayscale
        # img = cv2.resize(img, (32,32))

        sobelout = Image.new('L', (img.shape[1],img.shape[0]))                                       #empty image
        gradx = numpy.array(sobelout, dtype = float)
        grady = numpy.array(sobelout, dtype = float)
        sobelDirec = numpy.array(sobelout, dtype = float)

        sobel_x = [[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]]
        sobel_y = [[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]]

        width = img.shape[1]
        height = img.shape[0]

        for x in range(1, height-1):
            for y in range(1, width-1):
                px = (sobel_x[0][0] * img[x-1][y-1]) + (sobel_x[0][1] * img[x][y-1]) + \
                     (sobel_x[0][2] * img[x+1][y-1]) + (sobel_x[1][0] * img[x-1][y]) + \
                     (sobel_x[1][1] * img[x][y]) + (sobel_x[1][2] * img[x+1][y]) + \
                     (sobel_x[2][0] * img[x-1][y+1]) + (sobel_x[2][1] * img[x][y+1]) + \
                     (sobel_x[2][2] * img[x+1][y+1])
                py = (sobel_y[0][0] * img[x-1][y-1]) + (sobel_y[0][1] * img[x][y-1]) + \
                     (sobel_y[0][2] * img[x+1][y-1]) + (sobel_y[1][0] * img[x-1][y]) + \
                     (sobel_y[1][1] * img[x][y]) + (sobel_y[1][2] * img[x+1][y]) + \
                     (sobel_y[2][0] * img[x-1][y+1]) + (sobel_y[2][1] * img[x][y+1]) + \
                     (sobel_y[2][2] * img[x+1][y+1])

                gradx[x][y] = px
                grady[x][y] = py
                if(px>0):
                    sobelDirec[x][y] = scipy.arctan(py^2/px^2)

        for x in range(height):
            for y in range(width):
                if sobelDirec[x][y] >= 0 and sobelDirec[x][y]<(360*(1/16)) \
                    or sobelDirec[x][y] > (360*(15/16)) and sobelDirec[x][y] <= 360:
                    sobelDirec[x][y] = 360

                for i in range(1,15,2):
                    if sobelDirec[x][y] >= (360*(i/16)) and sobelDirec[x][y]<(360*(i/16)):
                        sobelDirec[x][y] = (2*i)/16

        feature_vector = []
        for x in range(0,4):
            x = x*(height/4)
            for y in range(0,4):
                y = y*(width/4)
                a = sobelDirec[x:x + height, y:y + width]
                for p in range(45,9*45, 45):
                    count = sum(a==p)
                    feature_vector.append(count)

        return feature_vector
        #img = cv2.imread('h.jpg', cv2.IMREAD_GRAYSCALE)
        #MAT_Grad(img)

    def run(self,image):
        print "Running MAT Gradient"
        image = self.scale(image)
        feature = self.findMATGrad(image)
        return feature
import cv2
import scipy.ndimage as ndi
import scipy
import numpy
import Image
import math
from Groep2.preprocessing import thinning, prepImage


#def Edge_Maps(img):
# load an color image in grayscale
img = cv2.imread('cenfura.jpg', cv2.IMREAD_GRAYSCALE)

#the preprocessor object
prepper = prepImage.PreProcessor()
img = prepper.bgSub(img)
binary = prepper.binarize(img)
thin = thinning.thinning(binary)

G = thin

sobelout = Image.new('L', thin.shape[:2])                                       #empty image
gradx = numpy.array(sobelout, dtype = float)
grady = numpy.array(sobelout, dtype = float)

sobel_x = [[-1,0,1],
           [-2,0,2],
           [-1,0,1]]
sobel_y = [[-1,-2,-1],
           [0,0,0],
           [1,2,1]]

width = thin.shape[1]
height = thin.shape[0]

for x in range(1, width-1):
    for y in range(1, height-1):
        px = (sobel_x[0][0] * G[x-1][y-1]) + (sobel_x[0][1] * G[x][y-1]) + \
             (sobel_x[0][2] * G[x+1][y-1]) + (sobel_x[1][0] * G[x-1][y]) + \
             (sobel_x[1][1] * G[x][y]) + (sobel_x[1][2] * G[x+1][y]) + \
             (sobel_x[2][0] * G[x-1][y+1]) + (sobel_x[2][1] * G[x][y+1]) + \
             (sobel_x[2][2] * G[x+1][y+1])
        py = (sobel_y[0][0] * G[x-1][y-1]) + (sobel_y[0][1] * G[x][y-1]) + \
             (sobel_y[0][2] * G[x+1][y-1]) + (sobel_y[1][0] * G[x-1][y]) + \
             (sobel_y[1][1] * G[x][y]) + (sobel_y[1][2] * G[x+1][y]) + \
             (sobel_y[2][0] * G[x-1][y+1]) + (sobel_y[2][1] * G[x][y+1]) + \
             (sobel_y[2][2] * G[x+1][y+1])
        gradx[x][y] = px
        grady[x][y] = py
sobeloutmag = scipy.hypot(gradx, grady)
sobeloutdir = scipy.arctan2(grady, gradx)

for x in range(width):
    for y in range(height):
        if (sobeloutdir[x][y]<22.5 and sobeloutdir[x][y]>=0) or \
           (sobeloutdir[x][y]>=157.5 and sobeloutdir[x][y]<202.5) or \
           (sobeloutdir[x][y]>=337.5 and sobeloutdir[x][y]<=360):
            sobeloutdir[x][y]=0
        elif (sobeloutdir[x][y]>=22.5 and sobeloutdir[x][y]<67.5) or \
             (sobeloutdir[x][y]>=202.5 and sobeloutdir[x][y]<247.5):
            sobeloutdir[x][y]=45
        elif (sobeloutdir[x][y]>=67.5 and sobeloutdir[x][y]<112.5)or \
             (sobeloutdir[x][y]>=247.5 and sobeloutdir[x][y]<292.5):
            sobeloutdir[x][y]=90
        else:
            sobeloutdir[x][y]=135


#show the image
cv2.imshow('image',sobeloutdir)
cv2.waitKey(0)
cv2.destroyAllWindows()
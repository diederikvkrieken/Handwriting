__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

#System modules
import sys, os
import cv2
import numpy as np
import time

#Own Modules
from Groep2.preprocessing import prepImage
from Groep2.preprocessing import thinning

#VARIABLES
alpha = 4

# load an color image in grayscale
img = cv2.imread('cenfura.jpg', cv2.IMREAD_GRAYSCALE)


#the preprocessor object
prepper = prepImage.PreProcessor()
img = prepper.bgSub(img)
binary = prepper.binarize(img)
thin = thinning.thinning(binary)

#Sum column and find CSC candidates.
column_sum = cv2.reduce(thin, 0, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32F)
CSC_columns = cv2.threshold(column_sum, 1.0, 1.0,cv2.THRESH_BINARY_INV)[1]
CSC_columns[0,0] = 1
CSC_columns[0,-1] = 1
CSC_columns = CSC_columns[0,:]
SC_columns = []

m= n= sum = 0
current_k = 0
k = 0
for csc in CSC_columns:
    #print csc
    if csc == 1:
        #print k - current_k
        if (k - current_k) <= 2:
            sum +=  k
            n += 1
            print "N: ", n
        else:
            if n > 0:
                print "FOUND SC"
                SC_columns.append(int(round(sum/n)))
            else:
                SC_columns.append(k)
            m += 1
            sum= n= 0
        current_k = k
        #print current_k
    k += 1

with_lines = thin.copy()
with_lines_step3 = thin.copy()
thin_height, thin_width = thin.shape

for x in range(0, thin_width):
    if CSC_columns[x] == 1:
        cv2.line(with_lines,(x,0),(x,thin_height -1),(1),1)

for x in SC_columns:
        cv2.line(with_lines_step3,(x,0),(x,thin_height -1),(1),1)

cv2.imshow("img", img)
cv2.imshow("binary", binary * 255)
cv2.imshow("thin", thin * 255)
cv2.imshow("with_lines", with_lines * 255)
cv2.imshow("with_lines_step3", with_lines_step3 * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

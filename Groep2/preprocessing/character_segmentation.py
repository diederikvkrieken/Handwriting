__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

import cv2
import numpy as np
from Groep2.preprocessing import prepImage
from Groep2.preprocessing import thinning
import time


# load an color image in grayscale
img = cv2.imread('cenfura.jpg', cv2.IMREAD_GRAYSCALE)


#the preprocessor object
prepper = prepImage.PreProcessor()
img = prepper.bgSub(img)
binary = prepper.binarize(img)
thin = thinning.thinning(binary)

start = time.time()
column_sum = cv2.reduce(thin, 0, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32F)
CSC_columns = cv2.threshold(column_sum, 1.0, 1.0,cv2.THRESH_BINARY_INV)[1]
end = time.time()
print end - start

with_lines = thin.copy()
thin_height, thin_width = thin.shape

for x in range(0, thin_width):
    if CSC_columns[0, x] == 1:
        cv2.line(with_lines,(x,0),(x,thin_height -1),(1),1)

cv2.imshow("img", img)
cv2.imshow("binary", binary * 255)
cv2.imshow("thin", thin * 255)
cv2.imshow("with_lines", with_lines * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
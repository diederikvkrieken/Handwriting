__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

import cv2
import numpy as np

# load an color image in grayscale
img = cv2.imread('stigui.jpg', cv2.IMREAD_GRAYSCALE)

# blur image a bit to prevent most speckles from noise
img = cv2.GaussianBlur(img,(3,3),0)

"""

cv2.imshow('original', img)





cv2.imshow('substracted', res)
cv2.imshow('blured_original', img)
cv2.imshow('blur', blurred_img)



# Binarize image with the Otsu method. Set object pixels to 0, background to zero
binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

contours = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
for cnt in contours:
    if cv2.contourArea(cnt)<500:
        cv2.drawContours(binary,[cnt],0,0)


# Use binary image as mask
combined = img * binary

cv2.imshow('combined', combined)
cv2.imshow('binary', binary * 255)
"""
"""
blurred_img = cv2.GaussianBlur(img,(55,55),0)
sub = np.float32(img) - np.float32(blurred_img )
img = np.uint8(cv2.normalize(sub,sub,0,255,cv2.NORM_MINMAX))
"""

# Binarize image with the Otsu method. Set object pixels to 0, background to zero
binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
binary2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 0)


binaryRes = binary & binary2

cv2.imshow("binary", binary)
cv2.imshow("binary2", binary2)
cv2.imshow("binaryres", binaryRes)


# derotate
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40,1)), None, None, 1)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (100,1)), None, None, 1)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,20)), None, None, 1)

cpy = binary.copy()

cnt = cv2.findContours(cpy, 0, 2)[0][0]


rect = cv2.minAreaRect(cnt)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, 255, 2)

cv2.imshow("binaryblob", binary)
cv2.imshow("box", img)

#derotate:
rows, cols = img.shape
rotation = rect[2]

print rotation

# sometimes rotation is 90 degrees off:
if abs(rotation) > 10:
    rotation += 90

M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
img = cv2.warpAffine(img, M, (cols,rows))
cv2.imshow("derotated", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
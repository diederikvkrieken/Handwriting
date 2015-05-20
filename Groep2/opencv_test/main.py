__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

import cv2
import numpy as np

# load an color image in grayscale
img = cv2.imread('rectore.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('original', img)

# blur image a bit to prevent most speckles from noise
img = cv2.GaussianBlur(img,(5,5),0)

blurred_img = cv2.GaussianBlur(img,(55,55),0)

sub = np.float32(img) - np.float32(blurred_img )
res = np.uint8(cv2.normalize(sub,sub,0,255,cv2.NORM_MINMAX))

cv2.imshow('substracted', res)
cv2.imshow('blured_original', img)
cv2.imshow('blur', blurred_img)



# Binarize image with the Otsu method. Set object pixels to 0, background to zero
binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

contours = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
for cnt in contours:
    if cv2.contourArea(cnt)<500:
        cv2.drawContours(binary,[cnt],-1,0,-1)

# Use binary image as mask
combined = img * binary

cv2.imshow('combined', combined)
cv2.imshow('binary', binary * 255)


cv2.waitKey(0)
cv2.destroyAllWindows()
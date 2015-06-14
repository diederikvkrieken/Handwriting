import cv2
import numpy as np
import math
from skimage.color import rgb2gray
from skimage.util import pad

def strucCharacteristics(img):
    feature_vector = []

    img = cv2.resize(img, (32,32))
    imgGray = rgb2gray(img)
    imgBW = np.where(img > np.mean(imgGray), 1.0, 0.0)
    imgBW = pad(imgBW, 1, mode='constant', constant_values=1)

    y_hist = imgBW.sum(axis=0)
    x_hist = imgBW.sum(axis=1)
    feature_vector.append(x_hist)
    feature_vector.append(y_hist)
    for k in range(0, 72):
        k = k * 5
        total = 0
        for i in range(1, 16):
            total = total + imgBW[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))]
        feature_vector.append(total)
    return feature_vector
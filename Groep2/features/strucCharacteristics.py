import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2gray
from skimage.util import pad

img = cv2.imread('n_processed.ppm', 0)

imgGray = rgb2gray(img)
imgBW = np.where(img > np.mean(imgGray),1.0,0.0)
imgBW = pad(imgBW,1, mode='constant', constant_values=1)

y_hist = imgBW.sum(axis=0)
x_hist = imgBW.sum(axis=1)

print x_hist
print y_hist

plt.hist(x_hist, bins=32)
plt.show()
plt.hist(y_hist, bins=32)
plt.show()
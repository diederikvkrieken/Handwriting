import cv2
import math
from skimage.util import pad



class StrucCharacteristics():
    def __init__(self):
        self.xSize = 32
        self.ySize = 32
        pass

    def scale(self, img):
        # Scale to Sizes in Greyscale
        img = cv2.resize(img, (self.xSize, self.ySize))
        return img

    def findStrucCharacteristics(self,image):
        feature_vector = []

        # img = cv2.resize(img, (32,32))
        # imgGray = rgb2gray(img)
        # imgBW = np.where(img > np.mean(imgGray), 1.0, 0.0)
        imgBW = pad(image, 1, mode='constant', constant_values=1)

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

    def run(self,image):
        image = self.scale(image)
        feature = self.findStrucCharacteristics(image)
        return feature

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
        temp_vector = []
        for k in range(0, 72):
            k = k * 5
            total = 0
            for i in range(1, 16):
                total = total + imgBW[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))]
            temp_vector.append(total)
        feature_vector.append(temp_vector)


         #Combine feature
        featureMerged = [item for sublist in feature_vector for item in sublist]

        return featureMerged

    def run(self,image):
        print "Running Struc Characteristics"
        image = self.scale(image)
        feature = self.findStrucCharacteristics(image)
        return feature
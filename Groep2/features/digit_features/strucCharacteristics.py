import cv2
import math
import numpy
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
        #imgBW = pad(image, 1, mode='constant', constant_values=1)

        #If you want to print the images
        #radial_out = numpy.zeros((imgBW.shape[0],imgBW.shape[1],1), numpy.float)
        #radial_in = numpy.zeros((imgBW.shape[0],imgBW.shape[1],1), numpy.float)

        y_hist = image.sum(axis=0)
        x_hist = image.sum(axis=1)
        feature_vector.append(x_hist)
        feature_vector.append(y_hist)

        temp_vector = []
        for k in range(0, 72):
            k = k * 5
            total = 0
            for i in range(0, 15):
                total = total + image[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))]
            temp_vector.append(total)

        feature_vector.append(temp_vector)

        temp_vector = []
        for k in range(0, 72):
            k = k * 5
            for i in range(0, 15):
                value = image[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))]
                if value:
                    #radial_in[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))] = value
                    break
            temp_vector.append(value)

        feature_vector.append(temp_vector)

        temp_vector =[]
        for k in range(0, 72):
            k = k * 5
            for i in range(15, 0,-1):
                value = image[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))]
                if value:
                    #radial_out[abs(16 - i * math.sin(k)), abs(16 + i * math.cos(k))] = value
                    break
            temp_vector.append(value)


        #feature_vector.append(radial_in)
        #feature_vector.append(radial_out)
        feature_vector.append(temp_vector)

         #Combine feature
        featureMerged = [item for sublist in feature_vector for item in sublist]

        return featureMerged

    def run(self,image):
        print "Running Struc Characteristics"
        image = self.scale(image)
        feature = self.findStrucCharacteristics(image)
        return feature
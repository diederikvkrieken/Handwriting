import cv2
import math

class MultiZoning():

    def __init__(self):
        pass

    def doMultiZoning(self,image):
        cuts = [(3, 1), (1, 3), (2, 3), (3, 2), (3, 3), (1, 4), (4, 1), (4, 4), (6, 1), (1, 6), (6, 2), (2, 6), (6, 6)]
        imgheight, imgwidth = image.shape[:2]

        feature_vector = []

        for n in cuts:
            x_cuts = n[0]
            y_cuts = n[1]
            height = int(imgheight / y_cuts)
            width = int(imgwidth / x_cuts)
            imgSize = height * width
            temp_vector = []
            for i in range(0, y_cuts):
                i = i * height
                for j in range(0, x_cuts):
                    j = j * width
                    a = image[j:j + width, i:i + height]
                    percentage_black_pixels = a.sum() / imgSize

                    if math.isnan(percentage_black_pixels):
                        temp_vector.append(0)
                    else:
                        temp_vector.append(percentage_black_pixels)

            feature_vector.append(temp_vector)

        featureMerged = [item for sublist in feature_vector for item in sublist]
        return featureMerged

    def run(self,image):
        print "Multi Zoning"
        feature = self.doMultiZoning(image)
        return feature
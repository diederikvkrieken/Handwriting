# Standard libararies
import cv2
import math

class HOG():
    def __init__(self):
       pass

    # Extracts HOG features from an image and returns those
    def performHOG(self, img):

        opencv_hog= cv2.HOGDescriptor((8,8), (8,8), (8,8), (8,8), 9)
        # print "Hogging away! "
        alpha = 200

        if img.shape[0] > alpha or img.shape[1] > alpha:
            print "WARNING: A segment is larger than our padding code. This segment will be scalled and we might lose information!"
            cv2.imshow('BW', img * 255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img = cv2.resize(img, (alpha, alpha))

        else:
            # TODO write this code to be more efficient. I do not like the if statements.
            padTB = int(math.floor((alpha - img.shape[0]) / 2))
            padLR = int(math.floor((alpha - img.shape[1]) / 2))

            if (img.shape[0] + padTB * 2) != 200:
                if (img.shape[1] + padLR * 2) != 200:
                    img = cv2.copyMakeBorder(img, padTB, padTB + 1, padLR, padLR + 1, cv2.BORDER_CONSTANT)
                else:
                    img = cv2.copyMakeBorder(img, padTB, padTB + 1, padLR, padLR, cv2.BORDER_CONSTANT)
            elif (img.shape[1] + padLR * 2) != 200:
                img = cv2.copyMakeBorder(img, padTB, padTB, padLR, padLR + 1, cv2.BORDER_CONSTANT)
            else:
                img = cv2.copyMakeBorder(img, padTB, padTB, padLR, padLR, cv2.BORDER_CONSTANT)


        assert img.shape[0] == 200 and img.shape[1] == 200

        h1 = opencv_hog.compute(img)
        #h2 = hog(img, orientations=4, pixels_per_cell=(4, 4), cells_per_block=(1, 1))

        #print len(h1) #, len(h2)

        ##fd, hog_image = hog(img, orientations=4, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=True)
        result = h1[:,0]
        #return [0.0]
        return result

    def run(self, img):
        # print "Hogging"
        return self.performHOG(img)
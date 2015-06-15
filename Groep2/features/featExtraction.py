"Skeleton for feature extraction"

# Standard libararies
import cv2
import math

# Import CSS
from scale_space import runCSS

# Import HOG
from skimage.feature import hog

# Import Digit Features
from digit_features import concavitiesMeasurement as cm

class Features():
    
    def __init__(self):
        pass
    
    # Extracts HOG features from an image and returns those
    def HOG(self, img):
        # image = cv2.imread('n_processed.ppm', 0) #reads in image, added for clarity
        # print "Hogging away! "

        alpha = 200
        # This is incredibly stupid we should fix this otherwise just fuck it
        # img = cv2.resize(img, (40, 40))

        if img.shape[0] > alpha or img.shape[1] > alpha:
            print "PROBLEM: A segment is larger than our padding code. This segment will be scalled and we might lose information!"
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

        fd, hog_image = hog(img, orientations=4, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=True)
        
        ''' can be used for plotting letters and input images
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            ax1.axis('off')
            ax1.imshow(image, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            
            #    Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            
            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            '''
        return fd
    
    # Extracts css features from an image and returns those
    def css(self, img):
        return runCSS.runCss().run(img)

    def concavitiesMeasurement(self, img):
        return cm.ConcavitiesMeasurement().run(img)

    # A cheapskate feature extraction that definitely yields vectors of equal length
    def cheapskate(self, img):
        return img.shape[0] # Yes, it returns the width of a segment! :D
"Skeleton for feature extraction"

from scale_space import runCSS

#Import HOG
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

from skimage.util import pad

class Features():
    
    def __init__(self):
        pass
    
    # Extracts HOG features from an image and returns those
    def hog(self, img):
        # image = cv2.imread('n_processed.ppm', 0) #reads in image, added for clarity
        print "Hogging away! ", img.shape

        # This is incredibly stupid we should fix this otherwise just fuck it
        # img = cv2.resize(img, (40, 40))

        if img.shape[0] > 200 or img.shape[1] > 200:
            print "PROBLEM: A segment is larger than our padding code."
            plt.figure('BW')
            plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
            plt.show()
            cv2.waitKey(0)

        img = cv2.copyMakeBorder(img,200-img.shape[0],200-img.shape[0],200-img.shape[1],200-img.shape[1],cv2.BORDER_CONSTANT)
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
    
    def dali(self):
        # This was not going to work, right?
        pass
    
    # Extracts css features from an image and returns those
    def css(self, img):
        return runCSS.runCss().run(img)

    # A cheapskate feature extraction that definitely yields vectors of equal length
    def cheapskate(self, img):
        return img.shape[0] # Yes, it returns the width of a segment! :D
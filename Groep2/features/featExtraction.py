"Skeleton for feature extraction"

from scale_space import runCSS

#Import HOG
from skimage.feature import hog

class Features():
    
    def __init__(self):
        pass
    
    # Extracts HOG features from an image and returns those
    def hog(self, img):
        # image = cv2.imread('n_processed.ppm', 0) #reads in image, added for clarity
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
        return runCSS.run(img)

    # A cheapskate feature extraction that definitely yields vectors of equal length
    def cheapskate(self, img):
        return img.shape[0] # Yes, it returns the width of a segment! :D
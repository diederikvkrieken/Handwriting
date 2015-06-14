"Skeleton for feature extraction"

from scale_space import runCSS


class Features():
    
    def __init__(self):
        pass
    
    # Extracts HOG features from an image and returns those
    def HOG(self,img):

    def dali(self):
        # This was not going to work, right?
        pass
    
    # Extracts css features from an image and returns those
    def css(self, img):
        print "CSS on the go! "
        return runCSS.runCss().run(img)

    # A cheapskate feature extraction that definitely yields vectors of equal length
    def cheapskate(self, img):
        return img.shape[0] # Yes, it returns the width of a segment! :D

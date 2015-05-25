"Skeleton for feature extraction"

from scale_space import runCSS

class Features():

    def __init__(self):
        pass

    # Extracts HoG features from an image and returns those
    def hog(self):
        #TODO
        pass

    def dali(self):
        # This was not going to work, right?
        pass

    # Extracts css features from an image and returns those
    def css(self, img):
        return runCSS.run(img)
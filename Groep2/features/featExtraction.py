"Skeleton for feature extraction"

# Standard libararies
import cv2
import math

# Import CSS
from scale_space import runCSS

# Import HOG
import hog

# Import Digit Features
from digit_features import concavitiesMeasurement as cm
from digit_features import Modified_Edge_Maps as em
from digit_features import multiZoning as mz
from digit_features import MAT_Gradient_Based as mat
from digit_features import strucCharacteristics as sc
from digit_features import imageProjections as ip

class Features():
    
    def __init__(self):
        # Dictionary of all classifiers 0 = blackwhite image, 1 = gray scale image!
        self.featureMethods = {'HOG': [hog.HOG(), 1],
                            'CSS': [runCSS.runCss(), 0],
                            'CM': [cm.ConcavitiesMeasurement(), 0],
                            'EM': [em.EdgeMaps(),1],
                            'MZ': [mz.MultiZoning(),0],
                            'MAT': [mat.MAT_Grad(),1],
                            'SC': [sc.StrucCharacteristics(),1],
                            'IP': [ip.ImageProjections(),0]}
        pass
    
    # Extracts HOG features from an image and returns those
    def HOG(self, img):
        return hog.HOG().run(img)
    
    # Extracts css features from an image and returns those
    def css(self, img):
        return runCSS.runCss().run(img)

    def concavitiesMeasurement(self, img):
        return cm.ConcavitiesMeasurement().run(img)

    # A cheapskate feature extraction that definitely yields vectors of equal length
    def cheapskate(self, img):
        return img.shape[0] # Yes, it returns the width of a segment! :D
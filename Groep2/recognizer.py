"""
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
"""

import sys

import cv2

from Groep2.preprocessing import prepImage
from Groep2.features import featExtraction
from Groep2.classification import classification


class Recognizer:
    """
    Recognizer class
    """

    def __init__(self):
        pass

    def main(self, ppm, inwords):
        # Read and preprocess
        prepper = prepImage.PreProcessor()  # Initialize preprocessor
        words = prepper.prep(ppm, inwords)

        # Debug show
        for word in words:
            cv2.imshow('Cropped word: %s' % word[1], word[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # # For reference if we want characters
        # words, chars = prepper.cropCV(prepper.orig, words)  # Crop words

        ## Feature extraction
        feat = featExtraction.Features()

        # Iterate through words to extract features
        features = []   # List containing all features
        classes = []    # List containing class (word) features belong to
        for word in words:
            features.append(feat.css(word[0]))
            classes.append(word[1])
        # NOTE: these are in order! Do not shuffle or you lose correspondence.
        # zip() is also possible of course, but I simply do not feel the need. :)

        #TODO this is a debug classification problem
        # features = range(100)
        # classes = [0] * 50 + [1] * 50

        ## Classification
        cls = classification.Classification()

        cls.fullPass(features, classes) # Just a full run

        ## results



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <image> <input .words file>" % sys.argv[0]
        sys.exit(1)
    Recognizer().main(sys.argv[1], sys.argv[2])
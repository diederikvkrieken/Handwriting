"""
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
"""

import sys

import cv2

from Groep2.preprocessing import prepImage


class Recognizer:
    """
    Recognizer class
    """

    def __init__(self):
        pass

    def main(self, ppm, words):
        # Read and preprocess
        prepper = prepImage.PreProcessor()  # Initialize preprocessor
        prepper.read(ppm)
        words, chars = prepper.cropCV(prepper.orig, words)  # Crop words
        # Debug print
        print "crops length: ", len(words)
        crop = words.pop()[0]
        cv2.imshow('test', words[9][0])
        cv2.waitKey(0)
        # pre-processing
        prepper.bgSub()
        prepper.binarize()
        # cv2.imshow('binarized', prepper.bw)
        # cv2.waitKey(0)

        # feature extraction


        # classification


        # results

        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <image> <input .words file>" % sys.argv[0]
        sys.exit(1)
    Recognizer().main(sys.argv[1], sys.argv[2])
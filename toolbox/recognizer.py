"""
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
"""

import sys
import prepImage
import numpy
import cv2


class Recognizer:
    """
    Recognizer class
    """

    def __init__(self):
        pass

    def main(self, ppm, words):
        prepper = prepImage.PreProcessor()
        img = cv2.imread(ppm)

        prepper.read(ppm)
        crops = prepper.cropCV(img, words)
        print "crops length: ", len(crops)

        # Testing to print crop
        #ppmfileout = open('tmp.ppm', 'wb')
        crop = crops.pop()[0]
        cv2.imshow('test', crops[9][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #ppmfileout.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <image> <input .words file>" % sys.argv[0]
        sys.exit(1)
    Recognizer().main(sys.argv[1], sys.argv[2])
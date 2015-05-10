"""
Pre-processes an image for handwriting recognition.
Reads in a .ppm image, binarizes to use as mask, apply on original
and outputs this as .ppm
"""

import wordio, pamImage

class PreProcessor:
    """
    Class used for pre-processing
    """

    def __init__(self):
        pass

    def read(self, inputPPM):
        # Reads the input file
        self.orig = pamImage.PamImage(inputPPM)

    def cut(self, inxml):
        # Read in a words xml
        reader = wordio.WordLayoutReader()
        lines, name = wordio.read(inxml)
        # Cut from image


    def binarize(self):
        # Binarizes the image
        for pixel in self.orig:
            if (pixel > threshold):
                # One, otherwise zero

    def output(self, outName):
        # Write to specified file
        open(outName)

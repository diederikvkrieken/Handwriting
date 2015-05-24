"""
Pre-processes an image for handwriting recognition.
Reads in a .ppm image, binarizes to use as mask, apply on original
and outputs this as .ppm
"""

from toolbox import wordio
import cv2
import numpy as np


class PreProcessor:
    """
    Class used for pre-processing
    """

    def __init__(self):
        pass

    def read(self, inputPPM):
        # Reads the input file
        self.orig = cv2.imread(inputPPM)

    # Crops an image based on a words xml
    def cropCV(self, image, inxml):
        lines, name = wordio.read(inxml)

        # Arrays of tuples (cropped images, text)
        words = []
        characters = []

        # Cut from image
        for line in lines:
            # Iterate over lines
            for word in line:
                # Add regions (words/characters) to respective arrays
                words.append((image[word.top:word.bottom,
                              word.left:word.right], word.text))
                for character in word.characters:
                    characters.append((image[character.top:character.bottom,
                                       character.left:character.right], character.text))

        # Return arrays
        return words, characters

    # Subtracts background from image
    def bgSub(self, img):
        pass

    # Binarizes image based on threshold
    def binarize(self):
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.orig, cv2.COLOR_BGR2GRAY)
        # Binarize, NOTE: Uses Otsu's method!
        (thresh, self.bw) = cv2.threshold(self.gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Obsolete method using provided code
    # def cut(self, inxml):
    #     # Read in a words xml
    #     lines, name = wordio.read(inxml)
    #
    #     # Keep track of where we are in the file
    #     line_iter = iter(lines)
    #     cur_line = line_iter.next()
    #     word_iter = iter(cur_line)
    #
    #     # Cut image
    #     crops = []  # Array of tuples (cropped images, text)
    #     for line in lines:
    #         # Iterate over lines
    #         for region in line:
    #             # Iterate over regions (words/characters)
    #             crops.append((croplib.crop(self.orig, region.left, region.top,
    #                                         region.right, region.bottom), region.text))
    #
    #     # Return array
    #     return crops
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

    # Reads a ppm as grayscale
    def read(self, inputPPM):
        # Reads the input file
        self.orig = cv2.imread(inputPPM, cv2.IMREAD_GRAYSCALE)

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
        # blur image a bit to prevent most speckles from noise
        img = cv2.GaussianBlur(img,(5,5),0)

        # Make bigger blur and subtract to get rid of background
        blur = cv2.GaussianBlur(img,(55,55),0)
        img = np.float32(img) - np.float32(blur)

        # return contrast stretched image
        return np.uint8(cv2.normalize(img,img,0,255,cv2.NORM_MINMAX))

    # Binarizes image and uses that as mask
    def binarize(self, img):
        # Binarize image with the Otsu method. Set object pixels to 1, background to zero
        (thr, binary) = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use binary image as mask
        return (img * binary)

    # Preprocesses provided image
    def prep(self, inimg, inxml):
        # Crop all words from image
        self.read(inimg)
        words, characters = self.cropCV(self.orig, inxml)

        # For all words, subtract background, binarize and multiply with original
        prossed = []    # New list for tuples because tuples are immutable...
        for w in words:
            pros = self.bgSub(w[0])
            pros = self.binarize(w[0])
            prossed.append((pros, w[1]))

        # Return pre-processed words
        return prossed


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
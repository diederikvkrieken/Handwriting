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

    # Binarizes image and uses that as mask. Returns an image with 0 as background and 255 as object
    def binarize(self, img):
        # Binarize image with the Otsu method. Set object pixels to 0, background to zero
        binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        binary2 = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 0)

        binaryRes = binary & binary2

        #find contours won't work good with border connected contours, so we use a 1 px border to disconnect them from the border
        binaryRes = cv2.copyMakeBorder(binaryRes, 1, 1, 1, 1, cv2.BORDER_CONSTANT, binaryRes, 255)

        (cnts, _) = cv2.findContours(binaryRes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.ones(binaryRes.shape[:2], dtype="uint8") * 255

        # loop over the contours
        for c in cnts:
            # if the contour is bad, draw it on the mask
            if cv2.contourArea(c) < 100:
                cv2.drawContours(mask, [c], -1, 0, -1)

        binaryRes = cv2.bitwise_and(binaryRes, binaryRes, mask=mask)

        #remove the border:
        rows, cols = binaryRes.shape
        binaryRes = binaryRes[1:rows-1, 1:cols-1]

        binary = binaryRes

        # return the resulting binary mask
        return (binaryRes)

    # Derotates all images in images list using a thresholded version
    def derotate(self, binary, images):

        # smear out the binary image to get one large blob
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40,1)), None, None, 1)
        # remove some extended parts of the big blob (the top of f's, bottom of p's etc)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (100,1)), None, None, 1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,20)), None, None, 1)

        cpy = binary.copy()
        cnt = cv2.findContours(cpy, 0, 2)[0][0]

        # Finds the surounding box of the image and it's rotation
        rect = cv2.minAreaRect(cnt)

        # Apply rotation to all images in images list
        new_list = []
        for image in images:
            rows, cols = image.shape
            rotation = rect[2]

            # Sometimes rotation is 90 degrees off:
            if abs(rotation) > 10:
                rotation += 90

            # Apply the rotation
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
            img = cv2.warpAffine(img, M, (cols,rows))

            new_list.append(img)
        return new_list

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
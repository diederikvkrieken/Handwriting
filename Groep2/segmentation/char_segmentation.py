__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

import cv2
import numpy as np
from skimage.measure import label

import compareSegments, thinning

class segmenter:
    '''
    Class for segmenting a word
    '''

    def __init__(self):
        # Parameters
        self.alpha = 4

    # Function to give the crops an annotation based on character annotations
    def annotate(self, cs_columns, annotations):
        compSeg = compareSegments.Comparator()
        return compSeg.compare(cs_columns, annotations)

    # Copied from prepImage, don't ask me what it does
    def ascender_descender(self, binary):

        # smear out the binary image to get one large blob
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40,1)), None, None, 1)

        # remove some extended parts of the big blob (the top of f's, bottom of p's etc)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (100,1)), None, None, 1)

        # BUG!!!!!!!!!!!!!!!!!!!!!!!! WHEN MORPHING RECTANGLE IS TO BIG THE BINARY WILL BECOME BLACK FILL IN FOR "ALPHAMORPH" EITHER 10 OR 20
        ALPHAMORPH = 5
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1,ALPHAMORPH)), None, None, 1)


        # cv2.imshow("CPY", binary*255)
        # cv2.waitKey(0)

        cpy = binary.copy()
        contourCNT = cv2.findContours(cpy, 0, 2)
        cnt = contourCNT[0][0]

        # Finds the bounding box of the image and it's rotation
        x,y,w,h = cv2.boundingRect(cnt)
        return (y, y+h)

    def crop_ascender(self, ascender_x, image):
        """
        This function crops the ascender part of the image, and then extends
        the resulting image border with whitespace until it has the size of the original image
        :param ascender_x:
        :param image:
        :return:
        """
        crop_img = image[0:ascender_x, 0:image.shape[1]]
        # add empty border
        crop_img = cv2.copyMakeBorder(crop_img, 0, image.shape[0] - crop_img.shape[0], 0, 0, cv2.BORDER_CONSTANT,value=0)
        return crop_img

    def crop_descender(self, descender_x, image):
        """
        This function crops the decender part of the image, and then extends
        the resulting image border with whitespace until it has the size of the original image
        :param descender_x:
        :param image:
        :return:
        """
        crop_img = image[descender_x:image.shape[0], 0:image.shape[1]]
        crop_img = cv2.copyMakeBorder(crop_img, image.shape[0] - crop_img.shape[0], 0, 0, 0, cv2.BORDER_CONSTANT,value=0)
        return crop_img


    def crop_sc_areas(self, SC_columns, asc, desc, image):
        """
        This function returns the crops coresponding to the SC_columns list
        It does some fancy stuff with connected components to keep the 'f' an 'g' alive.
        :param SC_columns:
        :param asc:
        :param desc:
        :param image:
        :return: the new SC_colums lists and the crops list
        """

        # storage for the results
        crop_list = []
        new_SC_columns = []

        # the coordinate of the previous SC_column. We start at x = zero
        prev_x = 0

        asc_crop = self.crop_ascender(asc, image)
        desc_crop = self.crop_descender(desc, image)

        # for every element in SC_columns, get the corresponding crop if there are
        # at all pixels in that crop. If so, add the crop to the crop list, and the x coordinate to the new SC columns list
        for x in SC_columns:
            crop  = image[0:image.shape[0], prev_x:x]
            extend_right_width = image.shape[1] - crop.shape[1] - prev_x
            crop = cv2.copyMakeBorder(crop, 0, 0, prev_x, extend_right_width, cv2.BORDER_CONSTANT,value=0)

            crop = cv2.bitwise_or(crop, asc_crop)
            crop = cv2.bitwise_or(crop, desc_crop)
            labeled = label(crop, 8, background=0)
            labeled_crop_mid = labeled[asc:desc, prev_x:x]
            labeled_crop_mid = np.array(labeled_crop_mid, dtype=np.uint8)


            min = np.min(labeled_crop_mid)
            max = np.max(labeled_crop_mid)


            if (max - min) != 0:
                hist = cv2.calcHist([labeled_crop_mid],[0],None,[256],[0, 256])
                mask = np.zeros(crop.shape[:2], dtype="uint8")
                for i in range(0  , max + 1):
                    if(hist[i] > 0):
                        in_range = cv2.inRange(labeled, i, i) / 255
                        mask = cv2.bitwise_or(in_range, mask)

                cpy = mask.copy()
                cnts = cv2.findContours(cpy, 0, 2)[0]
                if len(cnts) != 0:
                    x_start, __, width,__ = cv2.boundingRect(cnts[0])
                    for cnt in cnts:
                        xx_start,__,wwidth,__ = cv2.boundingRect(cnt)
                        if x_start > xx_start:
                            x_start = xx_start
                        if width < wwidth:
                            width = wwidth

                    # some visualisation
                    #cv2.line(mask,(x_start,0),(x_start, mask.shape[0] -1),(1),1)
                    # cv2.line(mask,(x_start + width,0),(x_start + width, mask.shape[0] -1),(1),1)

                    # get the definitive crop and add it to the list
                    def_crop = mask[0:mask.shape[1], x_start: x_start + width]
                    crop_list.append(def_crop)

                    # some visualisation, you may uncomment
                    cv2.imshow("mask", mask * 255)
                    cv2.imshow("def crop", def_crop * 255)
                    cv2.imshow("cropp", crop * 255)
                    cv2.waitKey(0)

                # add the current column to the new list
                new_SC_columns.append(x)
            else:
                # there are no blobs in the crop, so do not add it to the new SC columns list
                pass

            prev_x = x
        return (new_SC_columns, crop_list)

    def step3(self, list, threshold):
        """
        This function applies step three of the paper
        :param list:
        :param threshold:
        :return: returns a list with the SC columns
        """
        m= n= sum = 0
        current_k = 0
        k = 0
        SC_columns = []
        for csc in list:
            if csc == 1:
                if (k - current_k) <= threshold and k != (len(list) -1):
                    sum +=  k
                    n += 1
                else:
                    if n > 0:
                        SC_columns.append(int(round(sum/n)))
                    else:
                        SC_columns.append(k)
                    m += 1
                    sum= n= 0
                current_k = k
            k += 1
        return SC_columns

    # Segments a given word image
    def segment(self, word):
        # calculate the y coordinates of the ascender (e.g. top part f) and descender (e.g. bottom part g) lines
        ascender, descender = self.ascender_descender(word)

        # step 1  of paper
        thin = thinning.thinning(word)

        #Sum column and find CSC candidates. (step 2 of paper)
        column_sum = cv2.reduce(thin, 0, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32F)
        CSC_columns = cv2.threshold(column_sum, 1.0, 1.0,cv2.THRESH_BINARY_INV)[1]
        CSC_columns[0,0] = 1
        CSC_columns[0,-1] = 1
        CSC_columns = CSC_columns[0,:]

        # apply step 3 and store
        SC_columns = self.step3(CSC_columns, 2)

        return self.crop_sc_areas(SC_columns, ascender, descender, word)


        # # draw CSC's and CS's
        # with_lines = thin.copy()
        # with_lines_step3 = thin.copy()
        # thin_height, thin_width = thin.shape
        #
        # for x in range(0, thin_width):
        #     if CSC_columns[x] == 1:
        #         cv2.line(with_lines,(x,0),(x,thin_height -1),(1),1)
        # for x in SC_columns:
        #         cv2.line(with_lines_step3,(x,0),(x,thin_height -1),(1),1)
        # # end of drawing CSC's and CS's

        # # draw ascender and descender lines
        # asc_desc = word.copy()
        # cv2.line(asc_desc,(0,ascender),(asc_desc.shape[1] -1,ascender),(1),1)
        # cv2.line(asc_desc,(0,descender),(asc_desc.shape[1],descender),(1),1)

__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

import cv2
import numpy as np
from skimage.measure import label
import sys

import compareSegments, thinning

class segmenter:
    '''
    Class for segmenting a word
    '''

    def __init__(self):
        pass

    # Function to give the crops an annotation based on character annotations
    def annotate(self, cs_columns, annotations):
        compSeg = compareSegments.Comparator()
        return compSeg.compare(cs_columns, annotations)


    # Copied from prepImage, don't ask me what it does
    def ascender_descender(self, binary):

        # smear out the binary image to get one large blob
       # binary = cv2.copyMakeBorder(binary, 0, 0, 110, 110, cv2.BORDER_CONSTANT,value=0)

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (30,1)), None, None, 2)
        # remove some extended parts of the big blob (the top of f's, bottom of p's etc)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (100  ,1)), None, None, 1)


        contourFound = False
        if cv2.__version__[0] == '3':
            # OpenCV 3 has an extra first return value
            (__, cnts, _) = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            (cnts, _) = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) >= 1:
            # get largest contour
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            contourFound = True

        y = 0
        h = binary.shape[0]

        # Finds the bounding box of the image and it's rotation
        if contourFound:
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


    def crop_sc_areas(self, SC_columns, asc, desc, imageBinary, imageGrayscale, xStart):
        """
        This function returns the crops coresponding to the SC_columns list
        It does some fancy stuff with connected components to keep the 'f' an 'g' alive.
        IMPORTANT: DOES NOT WORK WITH OPENCV3!!!!!!!
        :param SC_columns:
        :param asc:
        :param desc:
        :param image:
        :param xStart: this is the value that should be added to the SC_columns result list.
                        this value should be the x-part that is removed from the orginal image
                        before entering this function.
        :return: the new SC_colums lists and the crops list
        """

        # storage for the results
        crop_list = []
        new_SC_columns = []

        # the coordinate of the previous SC_column. We start at x = zero
        prev_x = 0

        asc_crop = self.crop_ascender(asc, imageBinary)
        desc_crop = self.crop_descender(desc, imageBinary)

        # for every element in SC_columns, get the corresponding crop if there are
        # at all pixels in that crop. If so, add the crop to the crop list, and the x coordinate to the new SC columns list
        for x in SC_columns:
            crop  = imageBinary[0:imageBinary.shape[0], prev_x:x]
            extend_right_width = imageBinary.shape[1] - crop.shape[1] - prev_x

            crop = cv2.copyMakeBorder(crop, 0, 0, prev_x, extend_right_width, cv2.BORDER_CONSTANT,value=0)

            crop = cv2.bitwise_or(crop, asc_crop)
            crop = cv2.bitwise_or(crop, desc_crop)
            labeled = label(crop, 8, background=0)
            labeled_crop_mid = labeled[asc:desc, prev_x:x]
            labeled_crop_mid = np.array(labeled_crop_mid, dtype=np.uint8)

            min = 0
            max = 0
            if labeled_crop_mid.size > 0:
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

                if cv2.__version__[0] == '3':
                    # OpenCV 3 has an extra first return value
                    cnts = cv2.findContours(cpy, 0, 2)[1]
                else:
                    cnts = cv2.findContours(cpy, 0, 2)[0]
                if len(cnts) != 0:
                    x_start, y_start, width,height = cv2.boundingRect(cnts[0])
                    for cnt in cnts:
                        xx_start,yy_start,wwidth,hheight = cv2.boundingRect(cnt)
                        if x_start > xx_start:
                            x_start = xx_start
                        if width < wwidth:
                            width = wwidth
                        if yy_start < y_start:
                            y_start = yy_start
                        if height < hheight:
                            height = hheight

                    # some visualisation
                   # cv2.line(mask,(x_start,0),(x_start, mask.shape[0] -1),(1),1)
                   # cv2.line(mask,(x_start + width,0),(x_start + width, mask.shape[0] -1),(1),1)

                    # get the definitive crop and add it to the list
                    def_crop = mask[y_start:y_start + height, x_start: x_start + width]
                    def_crop_grayscale = imageGrayscale[y_start:y_start + height, x_start: x_start + width]
                    crop_list.append((def_crop, def_crop_grayscale))

                    if def_crop.shape[1] < 1:
                        pass

                    # some visualisation, you may uncomment
                    #cv2.imshow("mask", def_crop * 255)
                    #cv2.imshow("cropp", def_crop_grayscale)
                    #cv2.waitKey(0)

                    # add the current column to the new list
                    new_SC_columns.append(x + xStart)
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
        n= sum = 0
        current_k = 0
        k = 0
        SC_columns = []
        for csc in list:
            if csc == 1:
                if (k - current_k) <= threshold and k != (len(list) -1):
                    sum += k
                    n += 1
                else:
                    if n > 0:
                        SC_columns.append(int(round(sum/n)))
                    else:
                        SC_columns.append(k)
                    sum= n= 0
                current_k = k
            k += 1
        return SC_columns

    def step3_revisited(self, listOrig, threshold):
        """
        This function does the same as step 3, but acts on an list without 'zero' elements
        :param list:
        :param threshold:
        :return: returns a list with the SC columns
        """
        sum = n = 0
        SC_columns = []
        for k in range(len(listOrig) -1):
            if listOrig[k + 1] - listOrig[k] <= threshold:
                sum += listOrig[k]
                n += 1
            else:
                if n > 0:
                    SC_columns.append(int(round((sum + listOrig[k]) / (n + 1))))
                    sum = n = 0
                else:
                    SC_columns.append(listOrig[k])

        if n!=0:
            SC_columns.append(int(round((sum) / (n))))

        SC_columns.append(listOrig[len(listOrig) -1])
        return SC_columns


    # Segments a given word image
    def segment(self, wordBinary, wordGrayscale):

        # let's delete empty space first and remember how much x-axis pixels we removed from the left
        #   ----------
        #   |        |    ------
        #   |  abcd  | => |abcd|
        #   |        |    ------
        #   ----------

        cpy = wordBinary.copy()

        if cv2.__version__[0] == '3':
            # OpenCV 3 has an extra first return value
            cnts = cv2.findContours(cpy, 0, 2)[1]
        else:
            cnts = cv2.findContours(cpy, 0, 2)[0]

        x_start, y_start, width,height = cv2.boundingRect(cnts[0])
        x_end = x_start + width;
        for cnt in cnts:
            xx_start,yy_start,width,hheight = cv2.boundingRect(cnt)
            if xx_start < x_start:
                x_start = xx_start
            if yy_start < y_start:
                y_start = yy_start
            if x_end < (xx_start + width):
                x_end = xx_start + width
            if height < hheight:
                height = hheight

        # img[y: y + h, x: x + w]
        wordBinary = wordBinary[y_start:y_start + height, x_start: x_end]
        wordGrayscale = wordGrayscale[y_start:y_start + height, x_start: x_end]

        #cv2.imshow("grayscale", wordGrayscale)
        #cv2.waitKey(1)

        # calculate the y coordinates of the ascender (e.g. top part f) and descender (e.g. bottom part g) lines
        ascender, descender = self.ascender_descender(wordBinary)


        # step 1  of paper
        thin = thinning.thinning(wordBinary)

        #Sum column and find CSC candidates. (step 2 of paper)
        if cv2.__version__[0] == '3':
            # OpenCV 3 does not contain the deprecated cv module
            column_sum = cv2.reduce(thin, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        else:
            column_sum = cv2.reduce(thin, 0, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32F)
        CSC_columns = cv2.threshold(column_sum, 1.0, 1.0,cv2.THRESH_BINARY_INV)[1]
        CSC_columns[0,0] = 1
        CSC_columns[0,-1] = 1
        CSC_columns = CSC_columns[0,:]

        # apply step 3 and store
        SC_columns = self.step3(CSC_columns, 2)


        # # draw CSC's and CS's
        # with_lines = thin.copy()
        # with_lines_step3 = thin.copy()
        # with_lines_step3_revised = thin.copy()
        # thin_height, thin_width = thin.shape
        #
        # for x in SC_columns:
        #     cv2.line(with_lines_step3,(x,0),(x,thin_height -1),(1),1)

        step3_revisited_treshold = 10

        # print "SC BEFORE STEP 3: ", SC_columns
        SC_columns = self.step3_revisited(SC_columns, step3_revisited_treshold)
        # print "SC After STEP 3", SC_columns

        """
        if len(SC_columns) >= 2:
            if (SC_columns[len(SC_columns) -1] - SC_columns[len(SC_columns) -2]) < step3_revisited_treshold:
                SC_columns[len(SC_columns) -2] = SC_columns[len(SC_columns) -1]
                SC_columns.remove(len(SC_columns)-2)
        """

        """
        step4_oversegmenting_Threshold = 20

        newList = []
        for i in range(len(SC_columns)):
            if i != 0:
                if not ((SC_columns[i] - SC_columns[i-1]) < step4_oversegmenting_Threshold):
                    newList.append(SC_columns[i])
        """

        result = []
        for SC_column in SC_columns:
            if SC_column > step3_revisited_treshold:
                result.append(SC_column)

        if len(result) == 0:
            result.append(wordBinary.shape[1])

        SC_columns = result

        # for x in SC_columns:
        #     cv2.line(with_lines_step3_revised,(x,0),(x,thin_height -1),(1),1)
        #
        # cv2.imshow("segments", with_lines_step3 * 255)
        # cv2.imshow("segments revisited", with_lines_step3_revised * 255)
        # cv2.waitKey(0)
        #end of drawing CSC's and CS's

        return self.crop_sc_areas(SC_columns, ascender, descender, wordBinary, wordGrayscale, x_start)


        # # draw ascender and descender lines
        # asc_desc = word.copy()
        # cv2.line(asc_desc,(0,ascender),(asc_desc.shape[1] -1,ascender),(1),1)
        # cv2.line(asc_desc,(0,descender),(asc_desc.shape[1],descender),(1),1)

    # Segments a given word image
    def segmentXML(self, wordBinary, wordGrayscale, wordXML):

        # let's delete empty space first and remember how much x-axis pixels we removed from the left
        #   ----------
        #   |        |    ------
        #   |  abcd  | => |abcd|
        #   |        |    ------
        #   ----------

        cpy = wordBinary.copy()

        if cv2.__version__[0] == '3':
            # OpenCV 3 has an extra first return value
            cnts = cv2.findContours(cpy, 0, 2)[1]
        else:
            cnts = cv2.findContours(cpy, 0, 2)[0]

        x_start, y_start, width,height = cv2.boundingRect(cnts[0])
        x_end = x_start + width;
        for cnt in cnts:
            xx_start,yy_start,width,hheight = cv2.boundingRect(cnt)
            if xx_start < x_start:
                x_start = xx_start
            if yy_start < y_start:
                y_start = yy_start
            if x_end < (xx_start + width):
                x_end = xx_start + width
            if height < hheight:
                height = hheight

        # img[y: y + h, x: x + w]
        wordBinary = wordBinary[y_start:y_start + height, x_start: x_end]
        wordGrayscale = wordGrayscale[y_start:y_start + height, x_start: x_end]

        #cv2.imshow("grayscale", wordGrayscale)
        #cv2.waitKey(1)

        # calculate the y coordinates of the ascender (e.g. top part f) and descender (e.g. bottom part g) lines
        ascender, descender = self.ascender_descender(wordBinary)

        # CREATE sc columns with From XML coords here and take into account that the last letter sometimes is longer then the binary cut.
        SC_columns = []
        wordXMLCount = 0
        for letter in wordXML:
                if letter[1] > wordBinary.shape[1]:
                    SC_columns.append(wordBinary.shape[1])
                    break
                else:
                    SC_columns.append(letter[1])

                wordXMLCount += 1


        # for x in SC_columns:
        #     cv2.line(with_lines_step3_revised,(x,0),(x,thin_height -1),(1),1)
        #
        # cv2.imshow("segments", with_lines_step3 * 255)
        # cv2.imshow("segments revisited", with_lines_step3_revised * 255)
        # cv2.waitKey(0)
        #end of drawing CSC's and CS's


        cuts, chars = self.crop_sc_areas(SC_columns, ascender, descender, wordBinary, wordGrayscale, x_start)

        # print "Entering for loop: ", len(cuts), " SC: ", len(SC_columns), " WORDXML LEN: ", len(wordXML)
        # print "SC: ", SC_columns
        # print "CUTS: ", cuts

        letterCount = 0
        segs = []
        for letter in wordXML:
                if letterCount > len(cuts)-1:
                    break
                elif letter[1] > wordBinary.shape[1]:
                    segs.append([cuts[letterCount], letter[2]])
                    break
                else:
                    segs.append([cuts[letterCount], letter[2]])
                letterCount += 1
        # print "WORDXML", wordXML
        # print "SEGS: ", segs
        return segs, chars


        # # draw ascender and descender lines
        # asc_desc = word.copy()
        # cv2.line(asc_desc,(0,ascender),(asc_desc.shape[1] -1,ascender),(1),1)
        # cv2.line(asc_desc,(0,descender),(asc_desc.shape[1],descender),(1),1)

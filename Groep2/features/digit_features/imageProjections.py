from scipy import ndimage
import cv2
import numpy as np


def imageProjections(image):
    # imgwidth, imgheight = image.shape[:2]
    feature_vector = []

    #rotation angle in degree
    rotated = ndimage.rotate(image, 45)
    # save 45 degree diagonal projection histogram
    diagonal_hist_a = rotated.sum(axis=0)
    feature_vector.append(diagonal_hist_a)

    imgwidth, imgheight = rotated.shape


    rotated = ndimage.rotate(image, -45)
    # save -45 degree diagonal projection histogram
    diagonal_hist_b = rotated.sum(axis=0)
    feature_vector.append(diagonal_hist_b)

    # Take quadrant of the rotated image for correct partitioned for feature
    left = rotated[0:(imgwidth/2), 0:(imgheight/2) ]
    top = rotated[(imgwidth/2):imgwidth, 0:(imgheight/2) ]
    right = rotated[(imgwidth/2):imgwidth , (imgheight/2):imgheight ]
    down = rotated[0:(imgwidth/2), (imgheight/2):imgheight ]

    # Rerotate in for axis sum, euclidian distance is NOT used, simple sum of each row
    left = ndimage.rotate(left, 45)

    # cv2.imshow('left rotated', left)
    # cv2.waitKey(1)

    imgwidth, imgheight = left.shape

    # Rotating twice gives us double the image, retrieve original part
    left = left[ (imgheight/4):(3*imgheight/4), (imgwidth/2):imgwidth ]
    hist_left = left.sum(axis=0)

    # cv2.imshow('left quadrant', left)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # same as left partition.
    right = ndimage.rotate(right, 45)
    right = right[(imgheight/4):(3*imgheight/4), :(imgwidth/2)]
    hist_right = right.sum(axis=0)

    top = ndimage.rotate(top, -45)
    top = top[ (imgheight/4):(3*imgheight/4), (imgwidth/2):imgwidth ]
    hist_top= top.sum(axis=0)

    down = ndimage.rotate(down, -45)
    down = down[(imgheight/4):(3*imgheight/4), :(imgwidth/2)]
    hist_down = down.sum(axis=0)

    feature_vector.append(hist_left)
    feature_vector.append(hist_right)
    feature_vector.append(hist_top)
    feature_vector.append(hist_down)
    return feature_vector
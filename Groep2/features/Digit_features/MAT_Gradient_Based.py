import cv2
import numpy
import Image
from Groep2.preprocessing import thinning, prepImage

def MAT_Grad(img):
        # load an color image in grayscale
        img = cv2.resize(img, (32,32))

        #the preprocessor object
        prepper = prepImage.PreProcessor()
        img = prepper.bgSub(img)
        binary = prepper.binarize(img)
        thin = thinning.thinning(binary)


        sobelout = Image.new('L', (thin.shape[1],thin.shape[0]))                                       #empty image
        gradx = numpy.array(sobelout, dtype = float)
        grady = numpy.array(sobelout, dtype = float)
        gradup = numpy.array(sobelout, dtype = float)
        graddown = numpy.array(sobelout, dtype = float)


        sobel_x = [[-1,0,1],
                   [-2,0,2],
                    [-1,0,1]]
        sobel_y = [[1,2,1],
                    [0,0,0],
                    [-1,-2,-1]]

        width = thin.shape[1]
        height = thin.shape[0]

        for x in range(1, height-1):
            for y in range(1, width-1):
                px = (sobel_x[0][0] * thin[x-1][y-1]) + (sobel_x[0][1] * thin[x][y-1]) + \
                    (sobel_x[0][2] * thin[x+1][y-1]) + (sobel_x[1][0] * thin[x-1][y]) + \
                     (sobel_x[1][1] * thin[x][y]) + (sobel_x[1][2] * thin[x+1][y]) + \
                     (sobel_x[2][0] * thin[x-1][y+1]) + (sobel_x[2][1] * thin[x][y+1]) + \
                    (sobel_x[2][2] * thin[x+1][y+1])
                py = (sobel_y[0][0] * thin[x-1][y-1]) + (sobel_y[0][1] * thin[x][y-1]) + \
                    (sobel_y[0][2] * thin[x+1][y-1]) + (sobel_y[1][0] * thin[x-1][y]) + \
                    (sobel_y[1][1] * thin[x][y]) + (sobel_y[1][2] * thin[x+1][y]) + \
                    (sobel_y[2][0] * thin[x-1][y+1]) + (sobel_y[2][1] * thin[x][y+1]) + \
                    (sobel_y[2][2] * thin[x+1][y+1])

                gradx[x][y] = px
                grady[x][y] = py


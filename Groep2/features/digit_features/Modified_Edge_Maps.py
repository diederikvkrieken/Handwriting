import cv2
import numpy
from Groep2.preprocessing import thinning, prepImage

class EdgeMaps():
    def __init__(self):
        self.xSize = 25
        self.ySize = 25
        pass

    def findEdgeMaps(self,img):
        # load an color image in grayscale
        img = cv2.resize(img, (self.xSize,self.ySize))

        #the preprocessor object
        prepper = prepImage.PreProcessor()
        img = prepper.bgSub(img)
        binary = prepper.binarize(img)
        thin = thinning.thinning(binary)

        sobelout = numpy.zeros((img.shape[0],img.shape[1],1), numpy.float)                                         #empty image
        gradx = sobelout.copy()
        grady = sobelout.copy()
        gradup = sobelout.copy()
        graddown = sobelout.copy()

        sobel_x = [[-1,2,-1],
                   [-1,2,-1],
                    [-1,2,-1]]
        sobel_y = [[-1,-1,-1],
                    [2,2,2],
                    [-1,-1,-1]]

        sobel_up = [[-1,-1,2],
                   [-1,2,-1],
                  [2,-1,-1]]

        sobel_down = [[2,-1,-1],
                  [-1,2,-1],
                  [-1,-1,2]]

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

                pup = (sobel_up[0][0] * thin[x-1][y-1]) + (sobel_up[0][1] * thin[x][y-1]) + \
                    (sobel_up[0][2] * thin[x+1][y-1]) + (sobel_up[1][0] * thin[x-1][y]) + \
                    (sobel_up[1][1] * thin[x][y]) + (sobel_up[1][2] * thin[x+1][y]) + \
                    (sobel_up[2][0] * thin[x-1][y+1]) + (sobel_up[2][1] * thin[x][y+1]) + \
                    (sobel_up[2][2] * thin[x+1][y+1])
                pdown = (sobel_down[0][0] * thin[x-1][y-1]) + (sobel_down[0][1] * thin[x][y-1]) + \
                    (sobel_down[0][2] * thin[x+1][y-1]) + (sobel_down[1][0] * thin[x-1][y]) + \
                    (sobel_down[1][1] * thin[x][y]) + (sobel_down[1][2] * thin[x+1][y]) + \
                    (sobel_down[2][0] * thin[x-1][y+1]) + (sobel_down[2][1] * thin[x][y+1]) + \
                    (sobel_down[2][2] * thin[x+1][y+1])

                gradx[x][y] = max(px-2,0)
                grady[x][y] = max(py-2,0)
                gradup[x][y] = max(pup-2,0)
                graddown[x][y] = max(pdown-2,0)

        horizontal = []
        vertical = []
        upwards = []
        downwards = []
        thimage =[]
        for x in range(0,5):
            x = x*5
            for y in range(0,5):
                y = y*5

                thin_image = thin[x:x + 5, y:y + 5]
                percent_thin = thin_image.sum() / 125
                thimage.append(percent_thin)

                hor_image = gradx[x:x + 5, y:y + 5]
                percen_horizontal = hor_image.sum() / 125
                horizontal.append(percen_horizontal)
        
                ver_image = grady[x:x + 5, y:y + 5]
                percen_vertical = ver_image.sum() / 125
                vertical.append(percen_vertical)
        
                up_image = gradup[x:x + 5, y:y + 5]
                percen_upwards = up_image.sum() / 125
                upwards.append(percen_upwards)
        
                down_image = graddown[x:x + 5, y:y + 5]
                percen_downwards = down_image.sum() / 125
                downwards.append(percen_downwards)

        feature_vector = []
        feature_vector.append(thimage)
        feature_vector.append(horizontal)
        feature_vector.append(vertical)
        feature_vector.append(upwards)
        feature_vector.append(downwards)
         #Combine feature
        featureMerged = [item for sublist in feature_vector for item in sublist]

        return featureMerged
        #cv2.destroyAllWindows()
        #img = cv2.imread('h.jpg', cv2.IMREAD_GRAYSCALE)
        #Edge_Maps(img)

    def run(self,img):
        print "Running Modified Edge Maps"
        feature = self.findEdgeMaps(img)
        return feature



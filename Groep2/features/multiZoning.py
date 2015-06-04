image = cv2.imread('n_processed.ppm', 0)

cuts = [(3,1),(1,3),(2,3),(3,2),(3,3),(1,4),(4,1),(4,4),(6,1),(1,6),(6,2),(2,6),(6,6)]
imgwidth, imgheight = image.shape[:2]

feature_vector = []

for n in cuts:
	x_cuts = n[1]
	y_cuts = n[2]
	height = imgheight/y_cuts
	width = imgwidth/x_cuts
	imgSize = height*width

	for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = image.crop(box)
            zeroes <- a.count(0)
            percentage_black_pixels = (imgSize-zeroes)/imgSize
            feature_vector.append(percentage_black_pixels)

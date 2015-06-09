	#Own Modules
from Groep2.preprocessing import prepImage
from Groep2.preprocessing import thinning

#def Edge_Maps(img):
	# load an color image in grayscale
	img = cv2.imread('cenfura.jpg', cv2.IMREAD_GRAYSCALE)


	#the preprocessor object
	prepper = prepImage.PreProcessor()
	img = prepper.bgSub(img)
	binary = prepper.binarize(img)
	thin = thinning.thinning(binary)
	print thin
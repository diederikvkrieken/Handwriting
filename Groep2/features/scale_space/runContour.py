from contourpreprocess import findContour
from skimage import io

def run():
	import sys
	
	if len(sys.argv) != 2:
		print "Usage: %s <image>" % sys.argv[0]
		sys.exit(1)
	
	# Read image
	img = io.imread(sys.argv[1])
	
	# Find contour
	fc = findContour(img)
	fc.run(iAlpha=1000)
	
	#Print image
	fc.printImage()
		
if __name__ == '__main__':
    run()

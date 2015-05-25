import numpy as np

import matplotlib.pyplot as plt

from css import CurvatureScaleSpace
from contourpreprocess import findContour
import cv2


def run(img):
	
	# Find contour
	fc = findContour(img)
	fc.run(iAlpha=500)	

	#Rebuild contour array
	curve = np.array([fc.contoursNormalized[:,1], fc.contoursNormalized[:,0]])
	
	#Run css
	c = CurvatureScaleSpace()
	cs = c.generate_css(curve, curve.shape[1], 0.01)
	vcs = c.generate_visual_css(cs, 9)

	plt.figure('Sample Curve')
	plt.plot(curve[0,:], curve[1,:],color='r')
	
	plt.figure('CSS')
	plt.plot(vcs)
	
	plt.show()


if __name__ == '__main__':
	import sys

	if len(sys.argv) != 2:
		print "Usage: %s <image>" % sys.argv[0]
		sys.exit(1)

	run(cv2.imread(sys.argv[1]))

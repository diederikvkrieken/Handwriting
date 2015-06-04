import numpy as np

import matplotlib.pyplot as plt

from css import CurvatureScaleSpace
from contourpreprocess import findContour
import cv2
import timeit


def run(img):
	
	# Find contour
	start = timeit.default_timer()
	fc = findContour(img)
	fc.run()


	#Rebuild contour array
	curve = np.array([fc.contoursNormalized[:,1], fc.contoursNormalized[:,0]])

	stop = timeit.default_timer()

	print stop - start
	start = timeit.default_timer()
	#Run css
	c = CurvatureScaleSpace()
	cs = c.generate_css(curve, curve.shape[1], 0.01)

	stop = timeit.default_timer()
	print "Total: ", stop - start

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

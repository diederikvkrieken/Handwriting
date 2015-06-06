import numpy as np

import matplotlib.pyplot as plt

from css import CurvatureScaleSpace
from ContourFinder import findContour
import timeit


class runCss(object):


	def run(self, img):
		# Find contour
		# start = timeit.default_timer()

		fc = findContour(img)
		fc.run()


		# Rebuild contour array
		curve = np.array([fc.contoursNormalized[:, 1], fc.contoursNormalized[:, 0]])

		# stop = timeit.default_timer()

		# print stop - start
		# start = timeit.default_timer()
		# Run css
		c = CurvatureScaleSpace()
		cs = c.generate_css(curve, curve.shape[1], 0.01)

		cssFeatures = np.amax(cs, axis=0)

		print len(cssFeatures)
		# stop = timeit.default_timer()
		# print "Total: ", stop - start

		# vcs = c.generate_visual_css(cs, 9)
		# plt.figure('Sample Curve')
		# plt.plot(curve[0,:], curve[1,:],color='r')

		# plt.figure('CSS')
		# plt.plot(vcs)

		# plt.show()

		return cssFeatures




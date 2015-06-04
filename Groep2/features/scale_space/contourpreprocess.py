import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import measure
from skimage import io
from skimage.color import rgb2gray
from scipy import interpolate

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.util import pad

class findContour(object):
	def __init__(self, img_file):
		self.imgGray = rgb2gray(img_file)
		self.imgBW = np.where(self.imgGray > np.mean(self.imgGray),1.0,0.0)
		self.imgBW = pad(self.imgBW,1, mode='constant', constant_values=1)

		
	def contour(self, inImage):
		# Find contours at a constant value of 0.8
		self.contours = measure.find_contours(inImage, 0)

		# Connect contours (not needed since we padd the image)
		if self.contours[0][0][0] != self.contours[0][-1][0] or self.contours[0][0][1] != self.contours[0][-1][1]:
			print "WARNING: Contour is not connected!"
			self.contours[0] = np.vstack([self.contours[0], self.contours[0][0]])
				
	def interpolateContour(self, alpha=1000):
		
		# x and y polygons of the contour
		x = self.contours[0][:,0]
		y = self.contours[0][:,1]

		# Perform spline interpolation
		t = np.arange(x.shape[0], dtype=float)
		t /= t[-1]
		nt = np.linspace(0, 1, alpha)
		x1 = interpolate.spline(t, x, nt)
		y1 = interpolate.spline(t, y, nt)
		
		# Update the contours
		self.contours = np.dstack((x1,y1))
		#return np.dstack((x1,y1))[0]
		
		
	def normalizeContour(self, alpha=200):
		
		#Calculate how many points the contours has
		contSize = self.contours[0].shape[0] - 1
		
		#Calculate arc lenght between every point.
		Lk = np.zeros([contSize+2])
		
		Lk[0] = 0
		L = 0.0
		for n, contour in enumerate(self.contours[0]):
			if n != contSize:
				dist = np.linalg.norm(contour-self.contours[0][n+1])
				Lk[n+1] = Lk[n] + dist
				L += dist
			else:
				dist = np.linalg.norm(contour-self.contours[0][0])
				Lk[n+1] = Lk[n] + dist
				L += dist
		
		# Time array
		T = Lk / L
		
		self.contoursNormalized = np.zeros([alpha+1,2])
		
		j = 0.0
		for n, t in enumerate(T):
			if (j / alpha) < t:
				self.contoursNormalized[j] = self.contours[0][n-1]
				j += 1
			
			if (j / alpha) == 1:
				self.contoursNormalized[alpha] = self.contoursNormalized[0]
				break
			
	def printImage(self):
		# Display the image and plot all contours found
		fig, ax = plt.subplots()
		ax.imshow(self.imgGray, interpolation='nearest', cmap=plt.cm.gray)

		for n, contour in enumerate(self.contours):
			ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='b')
		
		ax.plot(self.contoursNormalized[:, 1], self.contoursNormalized[:, 0], linewidth=2, marker='x', color='r')

		ax.axis('image')
		ax.set_xticks([])
		ax.set_yticks([])
		plt.show()
	
	def run_alt(self, nAlpha =200, iAlpha=1000):
		# Find contour
		self.contour(self.imgBW)
	
		# Interpolate
		if self.contours[0].shape[0] < (nAlpha + 100):
			print "Interpolation needed"
			self.interpolateContour(iAlpha)
	
		#Normalize
		self.normalizeContour(nAlpha)

	def run(self, nAlpha =200, iAlpha=300):
		
		# Find contour
		self.contour(self.imgBW)
	
		#Normalize
		self.normalizeContour(alpha=30)
		self.contours[0] = self.contoursNormalized
		
		# Interpolate
		if self.contours[0].shape[0] < (nAlpha + 100):
			print "Interpolation needed"
			self.interpolateContour(iAlpha)
	
		#Normalize
		self.normalizeContour(nAlpha)
    
		"""
		#Calculate how many points the contours has
		contSize = self.contoursNormalized.shape[0] - 1
		
		for n, contour in enumerate(self.contoursNormalized):
			if n != contSize:
				print np.linalg.norm(contour-self.contoursNormalized[n+1])
			else:
				print np.linalg.norm(contour-self.contoursNormalized[0])
		"""
	


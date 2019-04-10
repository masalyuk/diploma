#http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0809/ORCHARD/restore_image.html
import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class MF(ADenoiser):
	def __init__(self, params=None):
		if params is None:
			self.covar = 1
			self.max_diff = 1
			self.weight_diff = 1
			self.iterations = 2
			ADenoiser.__init__(self, "MF")
		else:
			ADenoiser.__init__(self, "MF", params)

	def __str__(self):
		info 		= "name: " 			+ str(self.name)
		covar 		= "covar: " 		+ str(self.covar)
		max_diff 	= "max_diff: " 		+ str(self.max_diff)
		weight_diff = "weight_diff: "	+ str(self.weight_diff)
		iterations 	= "iterations: " 	+ str(self.iterations)

		return info + "\n" + covar + "\n" + max_diff + "\n" + weight_diff + "\n" + iterations

	def dif_color(self, val1, val2, type="GRAY"):
		if type == "RGB":
			# r = 0
			# g = 1
			# b = 2
			return cv2.cvtColor(val1, cv2.COLOR_RGB2GRAY) - cv2.cvtColor(val2, cv2.COLOR_RGB2GRAY)
		# return (val1[0]-val2[0])*0.2126+(val1[1]-val2[1])*0.7152+(val1[2]-val2[3])*0.0722
		return val1 - val2

	def swap_image(self, src, dst):
		tmp = src.copy()
		src = dst.copy()
		dst = tmp.copy()
		return (src, dst)

	def check_bound(self, w,h,x,y):
		if x > 0 and x < w and y < h and y > 0:
			return True
		else:
			return False

	def neighbors_func(self, val, x, y, img, max_diff):
		# for 4 neighbors
		vector_x = [1, -1]
		vector_y = [1, -1]

		diff = 0
		for shift_x in vector_x:
			if self.check_bound(img.shape[0], img.shape[1], x + shift_x, y):
				diff = diff + min(self.dif_color(val[0], img[x + shift_x, y],) ** 2, max_diff)

		for shift_y in vector_y:
			if self.check_bound(img.shape[0], img.shape[1], x, y + shift_y):
				diff = diff + min(self.dif_color(val[0], img[x, y + shift_y]) ** 2, max_diff)
		'''
		# The component of the potential due to the
		# difference between neighbouring pixel values.
		V_diff = 0
		if r > 0:
			V_diff = V_diff + min(dif_color(val, noisy_image(c, s)) ** 2, max_diff)
	
		if r < weight:
			V_diff = V_diff + min((val - buffer(r + 1, c, s)) ** 2, max_diff)
	
		if c > 1:
			V_diff = V_diff + min((val - buffer(r, c - 1, s)) ** 2, max_diff)
	
		if c < height:
			V_diff = V_diff + min((val - buffer(r, c + 1, s)) ** 2, max_diff)
		'''
		return diff

	# hidden count of values in one component
	def generate_all_color_values(self, hidden, components=1):
		colors = numpy.zeros((hidden, components))
		for component in range(components):
			for value in range(hidden):
				colors[value, component] = value

		return colors

	def restore_image(self, image):

		(weight, height) = image.shape

		noisy_image = image.copy()
		hidden_image = numpy.zeros(image.shape)
		(noisy_image, hidden_image) = self.swap_image(noisy_image, hidden_image)
		# This value is guaranteed to be larger than the
		# potential of any configuration of pixel values.
		V_max = (weight * height) * ((256) **  2 / (2 * self.covar) + 4 * self.weight_diff * self.max_diff)
		colors = self.generate_all_color_values(255, 1)
		for i in range(self.iterations):
			(noisy_image, hidden_image) = self.swap_image(noisy_image, hidden_image)
			# Vary each pixel individually to find the
			# values that minimise the local potentials.
			for r in range(weight):
				for c in range(height):
					V_local = V_max
					min_val = -1
					for val in colors:
						# The component of the potential due to the known data.
						V_data = self.dif_color(val, image[r, c]) ** 2 / (2 * self.covar)
						V_diff = self.neighbors_func(val, r, c, noisy_image, self.max_diff)
						V_current = V_data + self.weight_diff * V_diff

						if V_current < V_local:
							min_val = val
							print("test")
							V_local = V_current

					hidden_image[r, c] = min_val

		return hidden_image

	def get_name(self):
		return ADenoiser.get_name(self)

	# dataImage - class image
	def denoise(self, dataImage):
		(image, name) = self.get_img_name(dataImage)

		if self.params is not None:

			self.covar = self.params.covar
			self.max_diff = self.params.max_diff
			self.weight_diff = self.params.weight_diff
			self.iterations = self.params.iterations

		denoised_im = image.copy()
		for ch in range(3):
			denoised_im[:,:,ch] = self.restore_image(image[:,:,ch])

		if self.dImgs.get(name) is None:
			self.dImgs[name] = list()

		self.dImgs[name].append(Pair(denoised_im, self.params))

		return denoised_im






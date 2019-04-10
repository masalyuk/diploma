import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class Bilateral(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.diameter = 21
			self.sigma_i = 250
			self.sigma_s = 1
			ADenoiser.__init__(self, "bileteral")
		else:
			ADenoiser.__init__(self,"bileteral", params)

	def __str__(self):
		info 			= "name: "		+ str(self.name)
		diameter_str	= "diameter: "	+ str(self.diameter)
		sigma_i_str		= "sigma_i: "	+ str(self.sigma_i)
		sigma_s_str		= "sigma_s: "	+ str(self.sigma_s)

		return info + "\n" + diameter_str + "\n" + sigma_i_str + "\n" +  sigma_s_str

	def get_name(self):
		return ADenoiser.get_name(self)

	def __gaussian(self, x, sigma):
		return (1.0 / (2 * numpy.pi * (sigma ** 2))) * numpy.exp(-(x ** 2) / (2 * (sigma ** 2)))

	def __distance(self, x1, y1, x2, y2):
		return numpy.sqrt(numpy.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

	def denoise(self, dataImage):
		(image, name) = self.get_img_name(dataImage)

		if self.params is not None:
			self.diameter = self.params.diameter
			self.sigma_i  = self.params.sigma_i
			self.sigma_s  = self.params.sigma_s

		denoise_image = cv2.bilateralFilter(image.astype("uint8"), self.diameter, self.sigma_i, self.sigma_s)
		if self.dImgs.get(name) is None:
			self.dImgs[name] = list()

		self.dImgs[name].append(Pair(denoise_image, self.params))

		return denoise_image

	def __bilateral_filter(self,image, diameter, sigma_i, sigma_s):
		new_image = numpy.zeros(image.shape)
		for row in range(len(image)):
			for col in range(len(image[0])):
				wp_total = 0
				filtered_image = 0
				for k in range(diameter):
					for l in range(diameter):
						n_x = row - (diameter / 2 - k)
						n_y = col - (diameter / 2 - l)
						if n_x >= len(image):
							n_x -= len(image)
						if n_y >= len(image[0]):
							n_y -= len(image[0])
						gi = self.__gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
						gs = self.__gaussian(self.__distance(n_x, n_y, row, col), sigma_s)
						wp = gi * gs
						filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
						wp_total = wp_total + wp
				filtered_image = filtered_image // wp_total
				new_image[row][col] = numpy.round(filtered_image).astype("uint8")
		return new_image
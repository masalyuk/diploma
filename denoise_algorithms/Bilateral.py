import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class Bilateral(ADenoiser):
	def __init__(self,	params=None):

		if params is None:
			self.params = {}
			self.params["diameter"] = 21
			self.params["sigma_i"] = 3
			self.params["sigma_s"] = 1

		ADenoiser.__init__(self,"Bilateral", params)

	def get_name(self):
		return ADenoiser.get_name(self)

	def __gaussian(self, x, sigma):
		return (1.0 / (2 * numpy.pi * (sigma ** 2))) * numpy.exp(-(x ** 2) / (2 * (sigma ** 2)))

	def __distance(self, x1, y1, x2, y2):
		return numpy.sqrt(numpy.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

	def denoise(self, dataImage):

		denoise_image = self.__bilateral_filter(dataImage.astype("uint8").copy(), self.params["diameter"], self.params["sigma_i"], self.params["sigma_s"])
		return denoise_image

	def __bilateral_filter(self, image, diameter, sigma_i, sigma_s):
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
import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class Bilateral(ADenoiser):
	def __init__(self,	params=None):

		if params is None:
			self.params = {}
			self.params["diameter"] = 3

			self.params["sigma_i"] = 255
			self.params["sigma_s"] = 10
		else:
			self.params = params

		self.params["diameter"] = int(self.params["diameter"])
		self.params["sigma_i"] = int(self.params["sigma_i"])
		self.params["sigma_s"] = int(self.params["sigma_s"])
		ADenoiser.__init__(self,"Bilateral", self.params)

	def get_name(self):
		return ADenoiser.get_name(self)

	def __gaussian(self, x, sigma):

		return numpy.exp(-(x**2)/(2*sigma**2))

	def __distance(self, x1, y1, x2, y2):
		return numpy.sqrt(numpy.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

	def denoise(self, dataImage):
		#denoise_image = self.__bilateral_filter(dataImage.copy(), self.params["diameter"], self.params["sigma_i"], self.params["sigma_s"])
		denoise_image = cv2.bilateralFilter(src=dataImage.copy(),sigmaColor=self.params["sigma_i"], sigmaSpace=self.params["sigma_s"],d=self.params["diameter"])
		return denoise_image.astype("uint8")

	def __bilateral_filter(self, image, diameter, sigma_i, sigma_s):
		new_image = numpy.zeros(image.shape)
		rows = image.shape[0]
		cols = image.shape[1]
		print(cols)
		print(rows)
		for row in range(rows):
			for col in range(cols):
				wp_total = 0
				filtered_image = 0
				for k in range(diameter):
					for l in range(diameter):
						n_x = row - (diameter / 2 - k - 1)
						n_y = col - (diameter / 2 - l - 1)


						if n_x < 0:
							n_x = 0
						if n_y < 0:
							n_y = 0

						if n_x >= rows:
							n_x = rows - 1
						if n_y >= cols:
							n_y = cols - 1



						gi = self.__gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)


						gs = self.__gaussian(self.__distance(n_x, n_y, row, col), sigma_s)

						wp = gi * gs

						filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
						wp_total = wp_total + wp

				filtered_image = filtered_image // wp_total
				new_image[row][col] = numpy.round(filtered_image)

		return new_image
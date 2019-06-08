#https://github.com/pfchai/GuidedFilter # for check implementation
import numpy
import cv2
from cv2.ximgproc import guidedFilter
from denoise_algorithms.ADenoiser import *

class Guided(ADenoiser):
	def __init__(self, params=None):
		if params is None:
			self.params = {}
			self.params["radius"] = 2
			self.params["eps"] = 0.010000000000000002
		else:
			self.params = params

		self.params["radius"] = int(self.params["radius"]) # else cannot to json
		ADenoiser.__init__(self, "Guided", self.params)


	def get_name(self):
		return ADenoiser.get_name(self)

	def alg(self,i ,p ,r ,e ):

		mean_i = cv2.boxFilter(i, ksize=(r,r), ddepth=-1)
		mean_p = cv2.boxFilter(p, ksize=(r,r), ddepth=-1)

		corr_i = cv2.boxFilter(i*i,ksize=(r,r), ddepth=-1)
		corr_ip = cv2.boxFilter(i*p,ksize=(r,r), ddepth=-1)

		var_i = corr_i - mean_i * mean_i
		cov_ip = corr_ip - mean_i * mean_p

		a = cov_ip / (var_i + e)
		b = mean_p - a * mean_i

		mean_a = cv2.boxFilter(a,ksize=(r,r), ddepth=-1)
		mean_b = cv2.boxFilter(b, ksize=(r,r), ddepth=-1)

		return (mean_a*i+mean_b)


	def denoise(self, dataImage):
		image = dataImage.copy().astype("uint8")
		denoised_im = numpy.zeros_like(image)

		denoised_im = self.alg(image/255, image/255, self.params["radius"], self.params["eps"])
		return denoised_im * 255


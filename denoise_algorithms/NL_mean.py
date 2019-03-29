import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class NL_mean(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.template_size = 100
			self.search_size = 250
			ADenoiser.__init__(self, "NL_mean")
		else:
			ADenoiser.__init__(self,"NL_mean", params)

	def __str__(self):
		info 				= "name: "		+ str(self.name)
		template_size_str	= "template_size: "		+ str(self.template_size)
		search_size_str		= "search_size: "	+ str(self.search_size)

		return info + "\n" + template_size_str + "\n" + search_size_str

	def get_name(self):
		ADenoiser.get_name(self)

	def denoise(self, image):
		print(self)
		if self.params is not None:
			self.template_size = self.params.template_size
			self.search_size = self.params.search_size

		print("max:" + str(numpy.max(image)))
		print("min:" + str(numpy.min(image)))
		denoised_im = image.copy()

		return cv2.fastNlMeansDenoisingColored(image.astype("uint8"))


import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class NL_mean(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.params = {}
			self.params["template_size"] = 7
			self.params["search_size"] = 16
			self.params["similar"] = 20
		else:
			self.params = params

		ADenoiser.__init__(self,"NL_mean", self.params)

	def get_name(self):
		return ADenoiser.get_name(self)

	def denoise(self, dataImage):
		denoised_im = numpy.zeros_like(dataImage)
		denoised_im = cv2.fastNlMeansDenoisingColored(dataImage.astype("uint8").copy(),\
													  templateWindowSize=self.params["template_size"], \
													  searchWindowSize=self.params["search_size"],\
													  h=self.params["similar"])


		return denoised_im


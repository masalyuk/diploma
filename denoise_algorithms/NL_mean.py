import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class NL_mean(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.params = {}
			self.params["template_size"] = 7
			self.params["search_size"] = 7

		ADenoiser.__init__(self,"NL_mean", params)

	def get_name(self):
		return ADenoiser.get_name(self)

	def denoise(self, dataImage):
		denoised_im = numpy.zeros_like(dataImage)
		cv2.fastNlMeansDenoisingColored(dataImage.astype("uint8").copy(), dst=denoised_im,h=16,templateWindowSize=4,hColor=10)


		return denoised_im


import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class NL_mean(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.template_size = 7
			self.search_size = 7
			ADenoiser.__init__(self, "NL_mean")
		else:
			ADenoiser.__init__(self,"NL_mean", params)

	def __str__(self):
		info 				= "name: "		+ str(self.name)
		template_size_str	= "template_size: "		+ str(self.template_size)
		search_size_str		= "search_size: "	+ str(self.search_size)

		return info + "\n" + template_size_str + "\n" + search_size_str

	def get_name(self):
		return ADenoiser.get_name(self)

	def denoise(self, dataImage):
		(image, name) = self.get_img_name(dataImage)
		if self.params is not None:
			self.template_size = self.params.template_size
			self.search_size = self.params.search_size

		denoised_im = numpy.zeros_like(image)
		cv2.fastNlMeansDenoisingColored(image.astype("uint8"), dst=denoised_im,h=16,templateWindowSize=4,hColor=10)

		if self.dImgs.get(name) is None:
			self.dImgs[name] = list()

		self.dImgs[name].append(Pair(denoised_im, self.params))
		return denoised_im


import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class TVL1(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.iter = 100
			self.lyambda = 250
			ADenoiser.__init__(self, "TVL1")
		else:
			ADenoiser.__init__(self,"TVL!", params)

	def __str__(self):
		info 			= "name: "		+ str(self.name)
		iter_str		= "iter: "		+ str(self.iter)
		lymbda_str		= "lymbda: "	+ str(self.lyambda)

		return info + "\n" + iter_str + "\n" + lymbda_str

	def get_name(self):
		ADenoiser.get_name(self)

	def denoise(self, image):
		print(self)
		if self.params is not None:
			self.iter = self.params.iter
			self.lyambda = self.params.lyambda

		print("max:" + str(numpy.max(image)))
		print("min:" + str(numpy.min(image)))
		denoised_im = image.copy()
		cv2.denoise_TVL1(image.astype("uint8"),denoised_im, self.lyambda, self.iter)
		return denoised_im


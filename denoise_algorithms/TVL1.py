import numpy
import cv2
from denoise_algorithms.ADenoiser import *

class TVL1(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.iter = 1000
			self.lyambda = 100000
			ADenoiser.__init__(self, "TVL1")
		else:
			ADenoiser.__init__(self,"TVL!", params)

	def __str__(self):
		info 			= "name: "		+ str(self.name)
		iter_str		= "iter: "		+ str(self.iter)
		lymbda_str		= "lymbda: "	+ str(self.lyambda)

		return info + "\n" + iter_str + "\n" + lymbda_str

	def get_name(self):
		return ADenoiser.get_name(self)


	#image - class image
	def denoise(self, dataImage):
		(image, name) = self.get_img_name(dataImage)
		if self.params is not None:
			self.iter = self.params.iter
			self.lyambda = self.params.lyambda

		denoised_im = numpy.zeros_like(image)
		print(self)

		cv2.denoise_TVL1(observations=denoised_im, result=image)
		cv2.imwrite("lkkkkk.jpg", denoised_im)

		if self.dImgs.get(name) is None:
			self.dImgs[name] = list()

		self.dImgs[name].append(Pair(denoised_im, self.params))

		return denoised_im


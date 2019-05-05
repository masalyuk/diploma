import numpy
import cv2

from denoise_algorithms.ADenoiser import *

class TVL1(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.params = {}
			self.params["iter"] = 50
			self.params["lyambda"] = round(1.5, 3)
			ADenoiser.__init__(self, "TVL1", self.params)
		else:
			params["lyambda"] = round(params["lyambda"], 3)
			ADenoiser.__init__(self, "TVL1", params)



	def __str__(self):
		info 			= "name: "		+ str(self.name)
		iter_str		= "iter: "		+ str(self.iter)
		lymbda_str		= "lymbda: "	+ str(self.lyambda)

		return info + "\n" + iter_str + "\n" + lymbda_str

	def get_name(self):
		return ADenoiser.get_name(self)

	def nabla(seld,I):
		h, w = I.shape
		G = numpy.zeros((h, w, 2), I.dtype)
		G[:, :-1, 0] -= I[:, :-1]
		G[:, :-1, 0] += I[:, 1:]
		G[:-1, :, 1] -= I[:-1]
		G[:-1, :, 1] += I[1:]
		return G

	def nablaT(self, G):
		h, w = G.shape[:2]
		I = numpy.zeros((h, w), G.dtype)
		# note that we just reversed left and right sides
		# of each line to obtain the transposed operator
		I[:, :-1] -= G[:, :-1, 0]
		I[:, 1:] += G[:, :-1, 0]
		I[:-1] -= G[:-1, :, 1]
		I[1:] += G[:-1, :, 1]
		return I

	# little auxiliary routine
	def anorm(self,x):
		'''Calculate L2 norm over the last array dimention'''
		return numpy.sqrt((x * x).sum(-1))

	def project_nd(self, P, r):
		'''perform a pixel-wise projection onto R-radius balls'''
		nP = numpy.maximum(1.0, self.anorm(P) / r)
		return P / nP[..., numpy.newaxis]

	def shrink_1d(self, X, F, step):
		'''pixel-wise scalar srinking'''
		return X + numpy.clip(F - X, -step, step)

	def tv_denoise(self, image, clambda, iter_n):
		# setting step sizes and other params
		L2 = 8.0
		#tau = 0.02
		tau = 0.02
		sigma = 1.0 / (L2 * tau)
		theta = 1.0

		X = image.copy()/255.0
		P = self.nabla(X)
		for i in range(iter_n):
			P = self.project_nd(P + sigma * self.nabla(X), 1)
			X1 = self.shrink_1d(X - tau * self.nablaT(P), image/255.0, clambda * tau)
			X = X1 + theta * (X1 - X)

		return X*255

	#image - class image
	def disp_parameters(self):
		return "iter" + str(self.iter) + "_lyamd" + str(self.lyambda)


	def denoise(self, dataImage):

		denoised_im = numpy.zeros_like(dataImage)
		for ch in range(3):
			denoised_im[:,:,ch] = self.tv_denoise(dataImage[:,:,ch], self.params["lyambda"], self.params["iter"])

		return denoised_im


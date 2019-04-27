import numpy
import cv2

class Result:
	"""

	Class Result contais original image, path to original image,
	noised image, denoised image, var and noise reduction algorithm

	"""
	def __read_image__(self):
		""" read image during init"""
		self.orig_im = cv2.imwrite(self.path)

	def __init__(self, path, image, algorithm):
		self.path = path

		if image is None:
			self.__read_image__()
		else:
			self.orig_im = image.copy()

		self.nois_im = None
		self.denois_im = None
		self.alg = algorithm
		self.var = None
		self.psnr = -1

	def set_psnr(self, psnr):
		self.psnr = psnr

	def get_psnr(self):
		return self.psnr

	def get_alg(self):
		return self.alg

	def get_var(self):
		return self.var

	def get_noised_image(self):
		if self.nois_im is None:
			print("Result doesn't contain noised image")

		return self.nois_im

	def get_denoised_image(self):
		if self.denois_im is None:
			print("Result doesn't denoised image")
		else:
			return self.denois_im

	def get_original_image(self):
		if self.orig_im is None:
			print("Result doesn't contain original image")
		else:
			return self.orig_im

	def get_path(self):
		return self.path

	def set_noised_image(self, image, var=None):
		self.var = var

		if image is None:
			print("Seted image is null")
		else:
			self.nois_im = image.copy()

	def set_var(self, var):
		self.var = var

	def set_denoised_image(self, im):
		if im is None:
			print("Deoised image is null")
		else:
			self.denois_im = im.copy()
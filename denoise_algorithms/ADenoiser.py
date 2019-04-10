import numpy

class Pair:
	first=None
	second=None

	def __init__(self, f, s):
		self.f = f
		self.s = s

	def first(self):
		return self.f

	def second(self):
		return self.s

class ADenoiser:
	dImgs = None #dictionary of pair(image, param)
	name = "ADenoiser"
	params = None
	def __init__(self, name, params=None):
		self.name = name
		self.params = params
		self.dImgs = {}
		#self.set_parameters(self, params)

	def set_parameters(self, params):
		self.params = params

	def get_img_name(self, image):
		return (image.im_n, image.name)

	def get_name(self):

		print(self.name)
		return self.name

	def disp_parameters(self):
		pass

	def denoise(self, image):
		pass
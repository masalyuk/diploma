import numpy
class ADenoiser:
	"img - denoise image"
	name = "ADenoiser"
	params = None
	def __init__(self, name, params=None):
		self.name = name
		self.params = params
		#self.set_parameters(self, params)

	def set_parameters(self, params):
		self.params = params

	def get_name(self):
		print(self.name)

	def disp_parameters(self):
		pass

	def denoise(self, image):
		pass
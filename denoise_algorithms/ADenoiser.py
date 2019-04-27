import numpy
""" TODO ADD EQUAL OPERATOR. IT COMPARE NAME OF ALG AND PARAM"""
""" TODO ADD VALIDATE PARAM """
class ADenoiser:
	def __init__(self, name, params=None):
		self.name = name
		if params is None:
			self.params = {}
		else:
			self.params = params

	def set_parameters(self, params):
		self.params = params

	def get_name(self):
		return self.name

	def disp_parameters(self):
		pass

	def denoise(self, image):
		pass

	def __str__(self):
		info = "Name of alg: " + name + "\n"

		pairs = self.params.items()

		for pair in pairs:
			name_of_param = pair[0]
			value_of_param = pair[1]

			info += name_of_param + ": " + str(value_of_param) + "\n"

		return info

import cv2
class Image:
	def __init__(self, im, path):
		self.path = path
		self.im = im
		self.name = path.split("\\")[-1]

		if len(path.split("\\")) > 1:
			self.type = path.split("\\")[-2]
		else:
			print("ERROR: Type not found")

	def __init__(self, path):
		self.path = path
		self.im = cv2.imread(path)

		self.name = path.split("\\")[-1]

		if len(path.split("\\")) > 1:
			self.type = path.split("\\")[-2]
		else:
			print("ERROR: Type not found")

	def get_type(self):
		return self.type

	def get_name(self):
		return self.name

	def get_path(self):
		return self.path

	def get_image(self):
		return self.im
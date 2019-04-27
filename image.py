import cv2
class Image:
	def __init__(self, im, path):
		self.path = path
		self.im = im

	def __init__(self, path):
		self.path = path
		self.im = cv2.imread(path)

	def get_path(self):
		return self.path

	def get_image(self):
		return self.im
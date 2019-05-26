import numpy
import cv2
import json
import hashlib
import os
from os.path import join

class Result:
	"""

	Class Result contains original image, path to original image,
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
		self.ssim = -1

	def set_psnr(self, psnr):
		self.psnr = psnr

	def set_ssim(self, ssim):
		self.ssim = ssim

	def get_psnr(self):
		return self.psnr

	def get_ssim(self):
		return self.ssim

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

	def get_name_original_image(self):
		return self.path.split('\\')[1]

	def get_path(self):
		return self.path

	def set_noised_image(self, image, var=None):
		self.var = var

		if image is None:
			print("Setted image is null")
		else:
			self.nois_im = image.copy()

	def set_var(self, var):
		self.var = var

	def set_denoised_image(self, im):
		if im is None:
			print("Deoised image is null")
		else:
			self.denois_im = im.copy()

	def anyFiledsIsNull(self):
		pass

	def create_name_for_denoised_image(self, _dict):
		h_str = json.dumps(_dict)
		print("JSON:" + h_str)
		h = hashlib.md5()
		h.update(h_str.encode())
		print(h.hexdigest())
		return h.hexdigest()

	def save_result(self):

		self.write_json()
		self.write_denoised_images()

	def write_denoised_images(self):
		_path = self.get_path_for_res_optimal()

		#write noised image
		cv2.imwrite(_path + self.get_name_original_image(), self.get_noised_image())

		_den_im_dir = join(_path, "den_im")

		if os.path.exists(_den_im_dir) is False:
			os.mkdir(_den_im_dir)

		_dict = self.getUpDict()
		name_img = _dict.get('name_den')
		path_img = join(_den_im_dir, name_img)
		cv2.imwrite(path_img + ".png", self.get_denoised_image())

	def write_json(self):
		_path = self.get_path_for_res_optimal()
		# in _path contain folder json, folder with denoised_images,
		# folder with plot and noised images
		# images - because we have some img for test

		_json_dir = join(_path, "json")
		if os.path.exists(_json_dir) is False:
			os.mkdir(_json_dir)

		_dict = self.getUpDict()
		name_json = _dict.get('name_den')
		path_json = join(_json_dir, name_json) + ".json"

		fp = open(path_json, 'w')

		json.dump(_dict, fp)

	def get_path_for_res_optimal(self):
		# dict in res
		_dict = self.getUpDict()
		# folder with res of optimal param in dir optimal
		optimal_dir = "optimal"
		if os.path.exists(optimal_dir) is False:
			os.mkdir(optimal_dir)
		# in folder optimal exists folder with name alg
		_params = _dict.get('params')
		name_alg = _params.get('name')

		alg_dir = join(optimal_dir, name_alg)
		if os.path.exists(alg_dir) is False:
			os.mkdir(alg_dir)

		# in folder alg exits folder with diff variance
		_var = str(_dict.get('var'))
		var_dir = join(alg_dir, _var)

		if os.path.exists(var_dir) is False:
			os.mkdir(var_dir)

		return var_dir

	def getUpDict(self):
		#return dict with next keys
		# path to original image
		# var
		# psnr
		# keys from alg

		self.anyFiledsIsNull()

		cur_dict = {"path": self.path, "var": self.var, "params": self.alg.getUpDict()}
		name = self.create_name_for_denoised_image(cur_dict)
		cur_dict.update({"name_den": name})
		cur_dict.update({"psnr": self.psnr})
		cur_dict.update({"ssim": self.ssim})

		return cur_dict
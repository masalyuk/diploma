import numpy
import sys
import cv2
import os
import fnmatch
import math
import random

class Storage:
	"""

		Class Storage contain all results of denoising.
		It allows get filter result: alg, img, var

	"""

	def __init__(self, list_res = None):
		if list_res is None:
			self.l_res = []
		else:
			self.l_res = list_res.copy()
	
	def set_result(self, res):
		if res is not None:
			self.l_res.append(res)
	
	def get_results_by_name_of_alg(self, name):
		l_res_alg_name = []
	
		for res in self.l_res:
			if res.get_alg().get_name() == name:
				l_res_alg_name.append(res)
	
		return l_res_alg_name
	
	def get_results_by_alg(self, alg):
		l_res_alg = []
	
		for res in self.l_res:
			if res.get_alg() is alg:
				l_res_alg.append(res)

	
		return l_res_alg
	
	def get_results_by_image_and_var(self, path_image, var=None):
		l_res_im_var= []
	
		for res in self.l_res:
			if res.get_path() == path_image:
				if var is not None:
					if abs(var - res.get_var()) < 0.0001:
						l_res_im_var.append(res)
				else:
					l_res_im_var.append(res)

		return l_res_im_var
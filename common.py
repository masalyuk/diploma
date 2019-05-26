from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.util import  random_noise

from image import *
from result import *

from os.path import join
import json
import os
import numpy
import cv2
class Common:
	'''
	This class contain common function
	'''

	def __init__(self):
		self.dir_res = "optimal"
		self.lRes = []

	def PSNR(self, img, nimg):
		'''
		:param img:  original image
		:param nimg: noised image
		:return: return value of PSNR
		'''
		return psnr(img, nimg)

	def get_result(self):
		'''

		:return: list of class Results
		'''
		return self.lRes

	def SSIM(self, img, nimg):
		'''
		:param img:  original image
		:param nimg: noised image
		:return: return value of SSIM
		'''
		return ssim(img, nimg, multichannel=True)

	def add_noise_to_list(self):
		'''
		Add noise to list of images which contain in class Result
		:param lRes: list of result
		:return:
		'''
		for res in self.lRes:
			noised_img = self.add_noise(res.get_original_image(), res.get_var())
			res.set_noised_image(noised_img, res.get_var())

		return self.lRes

	def add_noise(self, img, variance):
		'''

		:param img:  original image
		:param variance:  variance of noise
		:return:
		'''
		return random_noise(img.copy() / 255, mode='gaussian', var=variance, clip=True) * 255

	def get_optimal_param(self, alg, variance, measure, path_im):
		'''
		This function read all json files which contain alg
		:param alg: name of algorithm denoising
		:param variance: noise
		:param measure: PSNR or SSIM
		:return: param of algorithm
		'''
		if measure is not "psnr" and measure is not "ssim":
			print("Unknown measure")

		name_alg = ""
		if type(alg) is str:
			name_alg = alg
		else:
			name_alg = alg.get_name()

		path = join(join(join(self.dir_res, name_alg), str(variance)), "json")
		lJson = self.read_json(path)

		best_measure = -10
		best_params = {}
		for dJson in lJson:
			if dJson["path"].split("\\")[-1] == path_im.split("\\")[-1]:
				val = dJson[measure]
				if val > best_measure:
					best_measure = val
					best_params = dJson["params"]

		return best_params

	def get_images(self, dir):
		'''

		:param dir: directory with images
		:return: all images in dir
		'''
		lImages = list()
		fImages = os.listdir(dir)

		for fImage in fImages:
			lImages.append(Image(join(dir,fImage)))

		return lImages

	def get_list_variance(self, bv=None, ev=None, sv=None):
		'''

		:param bv: first value of variance
		:param ev: last value of variance
		:param sv:
		:return: list of value of variances
		'''
		if bv is None:
			bv = 0.005
		if ev is None:
			ev = 1.0
		if sv is None:
			sv = 0.1

		return numpy.arange(start=bv, stop=ev, step=sv)

	def init_result(self, l_var, l_image, l_den):
		"""

		:param l_var: list of variance of gausiian noise
		:param l_image: list of class Image
		:param l_den: list of denoiser
		:return: list of class Result
		"""
		l_res = list()

		for den in l_den:
			for img in l_image:
				for var in l_var:
					res = Result(img.get_path(), img.get_image(), den)
					res.set_var(var)
					l_res.append(res)

		self.lRes = l_res.copy()
		return l_res.copy()

	def read_json(self, dir):
		'''

		:param dir: directory which contain all json files
		:return: list of dictt
		'''
		fJsons = os.listdir(dir)
		lJson = list()

		for fJson in fJsons:
			with open(join(dir, fJson)) as fj:
				json_str = fj.read()
				lJson.append(json.loads(json_str))

		return lJson

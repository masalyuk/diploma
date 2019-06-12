from typing import Union

import numpy
import sys
import cv2
import os
from os.path import join
import fnmatch
import math
import random
import json
import matplotlib.pyplot as plt
import shutil
import hashlib
from  skimage.util import  random_noise


from image import *
from result import *
from storage import *
from common import *

from denoise_algorithms.Bilateral import *
from denoise_algorithms.TVL1 import *
from denoise_algorithms.NL_mean import *
from denoise_algorithms.BM3D import *
from denoise_algorithms.MF import *
from denoise_algorithms.Guided import *
#############################
OPTIMAL_DIR	= "optimal"		#
RESULT_DIR	= "D:\\result"		#
TEST_DIR	= "test"		#
NOISED		= "noised"		#
DENOISED	= "denoised"	#
JSON		= "json"		#
IMAGE		= "image"		#
IMAGES		= "images"		#
#############################
def get_all_bm3d():
	l_BM3D = []
	l_BM3D.append(BM3D())
	return  l_BM3D.copy()

def get_all_nlmean():
	l_Nl = []
	lTmplSize = [3,5,7]
	lSrchSize = [5,7,11]
	lSim = [30,35,30]

	params = {}
	for t_s in lTmplSize:
		for s_s in lSrchSize:
			for sim in lSim:
				params["template_size"] = t_s
				params["search_size"] = s_s
				params["similar"] = sim
				l_Nl.append(NL_mean(params.copy()))

	return l_Nl.copy()

def get_all_guided():
	l_Guided = []

	lRad = numpy.arange(start=3, stop=8, step=2)
	lEps = [0.3,0.5,0.8]

	params = {}
	for r in lRad:
		for e in lEps:
			params["radius"] = r
			params["eps"] = e * e
			l_Guided.append(Guided(params.copy()))


	return l_Guided.copy()

def get_all_bilateral():
	l_Bilateral = []

	#lDiameters = numpy.arange(start=3, stop=9, step=2)
	lDiameters = [5,7,11,13]
	#lInten = numpy.arange(start=50, stop=255, step=20)
	lInten = [50,150,220]
	#lRange = numpy.arange(start=1, stop=9, step=2)
	lRange = [5, 7, 11]

	params = {}
	for diam in lDiameters:
		for sig_i in lInten:
			for sig_s in lRange:
				params["diameter"] = diam
				params["sigma_i"] = sig_i
				params["sigma_s"] = sig_s
				l_Bilateral.append(Bilateral(params.copy()))

	return l_Bilateral.copy()

def get_all_tv():
	l_Tv = []

	lLambda = [0.3, 0.6, 1]
	lIter = [60]

	params = {}
	for lamb in lLambda:
		for iter in lIter:

			params["iter"] = iter
			params["lyambda"] = lamb
			l_Tv.append(TVL1(params.copy()))

	return l_Tv.copy()
#Init all algorithms of denoise
def init_denoiser():
	lDenoiser = list()

	lDenoiser.extend(get_all_guided())
	lDenoiser.extend(get_all_nlmean())
	lDenoiser.extend(get_all_tv())
	#lDenoiser.extend(get_all_bm3d())
	lDenoiser.extend(get_all_bilateral())
	return lDenoiser




def get_im():#for part 2
	im_dir = "images"
	fIm = os.listdir(im_dir)
	return fIm


def has_dir(*dirs):
	return os.path.exists(join_dirs(*dirs))

def join_dirs(*dirs):
	path = ""
	for i in range(len(dirs)):
		path = join(path, str(dirs[i]))

	return path

def has_noised_image(dir, img, var):
	return has_dir(dir, img.get_type(), img.get_name(), str(var), NOISED, img.get_name())

def equal_json_and_params(path_to_json, den):
	with open(path_to_json) as fj:
		params_str = fj.read()
		params = json.loads(params_str)

		return den.equal_params(params)


def has_result_of_denoising(dir, den, img, var):
	path = join_dirs(dir, img.get_type(), img.get_name(), str(var), DENOISED, den.get_name(), JSON)
	create_dirs(dir, img.get_type(), img.get_name(), str(var), DENOISED, den.get_name())

	if has_dir(path) is True:
		for file in os.listdir(path):
			if equal_json_and_params(join_dirs(path, file), den) is True:
				return True

	return False

def load_noised_image_by_var(path):
	img=os.listdir(path)
	return cv2.imread(join_dirs(path, img[0]))

def load_noised_image(dir, img, var):
	type = img.get_type()
	name = img.get_name()
	path = join_dirs(dir, type, name, var, NOISED, name)

	return cv2.imread(path)

def get_noised_image(common, dir, img, var):
	if has_noised_image(dir, img, var) is False:
		noised_im = common.add_noise(img.get_image(), var)

		name = img.get_name()
		type = img.get_type()

		path_to_noised = join_dirs(dir, type, name, str(var), NOISED)
		create_dirs(dir, type, name, str(var), NOISED)

		path_to_noised_im = join(path_to_noised, name)
		cv2.imwrite(path_to_noised_im, noised_im)

		return noised_im
	else:
		return load_noised_image(dir, img, var)


def create_dirs(*dirs):
	path = ""
	for dir in dirs:
		path = join(path, dir)
		if os.path.exists(path) is False:
			os.mkdir(path)



def save_denoised_image(image_path, name, den_img):
	path_to_image = join(image_path, name)
	cv2.imwrite(path_to_image + ".png", den_img)

def save_params_of_denoising(json_path, name, den, measure=None):
	path_to_json = join(json_path, name)

	fp = open(path_to_json + ".json", 'w')
	json.dump(den.disp_parameters(), fp)

def create_name_for_denoised_image(_dict):
	h_str = json.dumps(_dict)

	h = hashlib.md5()
	h.update(h_str.encode())

	return h.hexdigest()

def save_result(img, var, den, den_img, measure=None):
	type = img.get_type()
	name = img.get_name()

	common_path = join_dirs(RESULT_DIR, type, name, str(var), DENOISED, den.get_name())

	create_dirs(RESULT_DIR, type, name, str(var), DENOISED, den.get_name())
	create_dirs(RESULT_DIR, type, name, str(var), DENOISED, den.get_name(),JSON)
	create_dirs(RESULT_DIR, type, name, str(var), DENOISED, den.get_name(),IMAGE)

	json_path = join_dirs(common_path, JSON)
	image_path = join_dirs(common_path, IMAGE)

	res_name = create_name_for_denoised_image(den.disp_parameters())
	save_denoised_image(image_path, res_name, den_img)
	save_params_of_denoising(json_path, res_name, den)


def denoise_and_save_result(common, dir, lAlg, lImg, lVar):
	all_nums = len(lAlg) * len(lImg) * len(lVar)
	all_c =  len(lAlg) * len(lImg) * len(lVar)
	for img in lImg:
		for alg in lAlg:
			for var in lVar:
				print("Denoise:" + alg.get_name())
				print("all: " + str(all_nums))
				print("curr: " + str(all_nums - all_c))
				all_c-=1

				if has_result_of_denoising(dir, alg, img, var) is False:
					noised_image = get_noised_image(common, dir, img, var)
					denoised_image = alg.denoise(noised_image)

					save_result(img, var, alg, denoised_image)

def init_images_by_type(type):
	lImgs = []
	images = os.listdir(join_dirs(IMAGES, type))

	for image in images:
		lImgs.append(Image(join_dirs(IMAGES, type, image)))

	return lImgs

def init_images():
	lImgs = []
	types = os.listdir(IMAGES)
	for type in types:
		lImgs.extend(init_images_by_type(type))

	return lImgs


def get_noised_images(dir, img, var):
	return load_noised_image(dir, img, var)

#dir C:\Users\nikit\PycharmProjects\diploma\result\architecture\141015071_740c33975f.jpg\0.05\denoised\Guided\json
def get_denoised_images_with_param_by_alg_and_fix_param(dir, alg, *fixed_params):
	lImages_with_param = []

	path_to_alg = join_dirs(dir, DENOISED, alg)
	path_to_json = join_dirs(path_to_alg, JSON)
	path_to_image = join_dirs(path_to_alg, IMAGE)

	for file in os.listdir(path_to_json):
		with open(join_dirs(path_to_json, file)) as fj:
			params_str = fj.read()
			params = json.loads(params_str)

			for prms in fixed_params:
				if params[prms[0]] == prms[1]:
					path = join_dirs(path_to_image, file.split(".")[0])

					lImages_with_param.append((path+".png", params))

	return lImages_with_param

def sort_result(result, flow_param):


	tmp = -1
	size = len(result)
	sorted(result, )
	for i in range(size):
		for j in range(size-1):
			if result[j][1][flow_param] > result[j+1][1][flow_param]:
				tmp = result[j]
				tmp1 = result[j+1]
				result[j + 1] = tmp
				result[j] = tmp1



	return result

def get_x(result, flow_param):
	x = []

	for res in result:
		x.append(res[1][flow_param])

	return x


def get_all_image_by_type(folder, type_im):
	type_dir = join_dirs(folder, type_im)
	return os.listdir(type_dir)

def get_all_types_by_dir(dir):
	return os.listdir(dir)

#fip Fixed parameters name : value
#flp Flowed parameter
def one_type_one_alg_one_var(common, folder, type_im, alg, var, fip, flp, measure="PSNR", mode="average"):

	dir_res = join_dirs(folder, type_im)
	lImgs = get_all_image_by_type(folder, type_im)

	plt.title("type:" + type_im + " var:" + str(var) + alg.get_name() + " " + str(fip))
	all_measure = numpy.zeros(0)
	X = 0
	for img in lImgs:
		dir_with_type = join_dirs(dir_res, img, var)

		image_path = join_dirs(IMAGES, type_im, img)

		imgs_and_param = get_denoised_images_with_param_by_alg_and_fix_param(dir_with_type, alg.get_name(), fip)

		if all_measure.shape[0] == 0:
			all_measure.resize(len(imgs_and_param))

		psnrs = []
		result = sort_result(imgs_and_param, flp)
		for img_and_param in result:
			path_to_img = img_and_param[0]

			d_im = cv2.imread(path_to_img) # denoised image
			im = cv2.imread(image_path) # original image

			if measure=="PSNR":
				psnrs.append(common.PSNR(im, d_im))
			else:
				psnrs.append(common.SSIM(im, d_im))


		na_psnrs = numpy.asarray(psnrs)
		all_measure += na_psnrs
		x  = get_x(result, flp)
		X = x

		if mode == "all":
			plt.ylabel(measure)
			plt.xlabel(flp)
			print(img)
			plt.legend(img)
			plt.plot(x, psnrs, label=img)

	if mode=="average":
		plt.ylabel(measure)
		plt.xlabel(flp)
		plt.plot(x, all_measure/len(lImgs), label=img)
	else:
		plt.legend(lImgs)

	plt.show()


def plot_for_guided(common, dir_res, type_im, var, measure, mode):

	name_fix_param = "radius"
	value_fix_pram = 0
	flow_param = ""
	if name_fix_param == "eps":
		value_fix_pram = 0.04000000000000001
		flow_param = "radius"
	else:
		name_fix_param = "radius"
		value_fix_pram = 7
		flow_param = "eps"
	one_type_one_alg_one_var(common, dir_res, type_im, Guided(), var,
								 measure=measure, fip=(name_fix_param, value_fix_pram),
							 		flp=flow_param, mode=mode)

def plot_for_tv(common, dir_res, type_im, var, measure, mode):

	name_fix_param = "iter"
	value_fix_pram = 0
	flow_param = ""
	if name_fix_param == "lyambda":
		value_fix_pram = 0.04000000000000001
		flow_param = "iter"
	else:
		name_fix_param = "iter"
		value_fix_pram = 60
		flow_param = "lyambda"
	one_type_one_alg_one_var(common, dir_res, type_im, TVL1(), var,
								 measure=measure, fip=(name_fix_param, value_fix_pram),
							 		flp=flow_param, mode=mode)



def find_max_measure(common, path_to_result, img, var, measure="PSNR"):
	path_to_denoised_images = join_dirs(path_to_result, IMAGE)

	max = -1
	max_img = ""
	val = -1
	path_to_orig = join_dirs(*tuple(path_to_result.replace("result","images").split("\\")[1:4]))
	orig_image = cv2.imread(path_to_orig)
	for img in os.listdir(path_to_denoised_images):
		path_to_image = join_dirs(path_to_denoised_images, img)
		if measure=="PSNR":
			val = common.PSNR(cv2.imread(path_to_image), orig_image)
		else:
			val = common.SSIM(cv2.imread(path_to_image), orig_image)

		if val > max:
			max = val
			max_img  = img


	return max


def find_max_alg(common, path_to_image_with_var, img,  var, measure="PSNR", class_image=None):
	path_to_alg = join_dirs(path_to_image_with_var, DENOISED)
	algs = os.listdir(path_to_alg)
	max = -1
	for alg in algs:
		if alg == "BM3D":
			continue
		path_to_result = join_dirs(path_to_alg, alg)
		val = find_max_measure(common, path_to_result, img, var, measure)

		if val > max:
			max = val
			max_alg = alg

	#check psnr or ssim with noised image
	path_to_noised = join_dirs(path_to_image_with_var, NOISED, img)

	orig_image  = cv2.imread(join_dirs(RESULT_DIR, class_image, img))
	noised_image = cv2.imread(path_to_noised)
	if measure == "PSNR":
		val = common.PSNR(noised_image, orig_image)
	else:
		val = common.SSIM(noised_image, orig_image)

	if val > max:
		max = val
		max_alg = "WO alg"
	return max_alg

def count_res(common, dir_res, class_image, var, measure="PSNR"):
	path_to_images = join_dirs(dir_res, class_image)
	res_algs = {}
	imgs = os.listdir(path_to_images)

	for img in imgs:
		name_alg = find_max_alg(common, join_dirs(path_to_images, img, str(var)),img,var, measure,class_image=class_image)
		if res_algs.get(name_alg) is None:
			res_algs[name_alg] = 1
		else:
			res_algs[name_alg] = res_algs[name_alg] + 1

	print(res_algs)

def main():
	common = Common()
	dir_res = RESULT_DIR

	lAlg = init_denoiser()
	lVar = [0.0001,0.005, 0.055, 0.15, 0.6]
	lImg = init_images()
	lMeasures = ["PSNR", "SSIM"]
	cats = ["architecture", "night", "person"]
	denoise_and_save_result(common, dir_res, lAlg, lImg, lVar)

	for cat in cats:
		for var in lVar:
			for meas in lMeasures:
				print("For: " + cat + " with " + str(var))
				print(meas)
				count_res(common,dir_res, cat, str(var), meas)
	#mode = "all" or "average"
	#measure = "PSNR" or "SSIM"
	#type_im = "night"
	#measure = "PSNR"
	#mode = "all"
	#alg_plot = "Guided"
	#if alg_plot == "Guided":
	#	plot_for_guided(common, dir_res, type_im, lVar[0], measure=measure, mode=mode)
	#elif(alg_plot == "TV"):
		#plot_for_tv(common, dir_res, type_im, lVar[0], measure=measure, mode=mode)

	#
	#
	# dir_with_type = join_dirs(dir_res, lImg[0].get_type(), lImg[0].get_name(), lVar[0])
	#
	# fixed_param = ("radius", 3)
	# flow_param = "eps"
	# imgs_and_param = get_denoised_images_with_param_by_alg_and_fix_param(dir_with_type, lAlg[0].get_name(), fixed_param)
	#
	# psnrs = []
	# result = sort_result(imgs_and_param, flow_param)
	# for img_and_param in result:
	# 	path_to_img = img_and_param[0]
	#
	# 	d_im = cv2.imread(path_to_img) # denoised image
	# 	im = lImg[0].get_image()
	#
	# 	psnrs.append(common.PSNR(im, d_im))
	#
	#
	# x  = get_x(result, flow_param)
	# plt.plot(x,psnrs)
	# plt.show()


if __name__ == "__main__":
	sys.exit(main())
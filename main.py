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
from  skimage.util import  random_noise

from image import *
from result import *
from storage import *

from denoise_algorithms.Bilateral import *
from denoise_algorithms.TVL1 import *
from denoise_algorithms.NL_mean import *
from denoise_algorithms.BM3D import *
from denoise_algorithms.MF import *
from denoise_algorithms.Guided import *

def add_gauss_noise(img, variance):
	return random_noise(img.copy()/255, mode='gaussian', var=variance, clip=True) * 255

def PSNR_list(img, list_img_noise):
	list_psnr = list()

	for noise_img in list_img_noise:
		list_psnr.append(PSNR(img, noise_img))

	return list_psnr

def PSNR(img, noise_img):
	mse = numpy.mean((img - noise_img) ** 2)
	if mse == 0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def init_var(b=0, e=1, s=0.1):
	l_var = list()
	l_var =numpy.arange(start=b, stop=e, step=s)
	return l_var

def init_images():
	lImages = list()

	imgs = os.listdir("images/")
	i = 0
	for img in imgs:
		lImages.append(Image(join("images", img)))
		i = i + 1
		break
		print("This image is " + str(i))
	return lImages

#Init all algorithms of denoise
def init_denoiser():
	lDenoiser = list()

	b = Bilateral()
	#lDenoiser.append(b)

	tv = TVL1()
	lDenoiser.append(tv)
	#params = {}
	#params["iter"] = 60
	#params["lyambda"] = 1.2
	#lDenoiser.append(TVL1(params=params))
	nl = NL_mean()
	#lDenoiser.append(nl)

	mf = MF()
	#lDenoiser.append(mf)

	g = Guided()
	#lDenoiser.append(g)

	bm3d = BM3D()
	#lDenoiser.append(bm3d)

	return lDenoiser


def init_result(l_var, l_image, l_den):
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

	return l_res

def add_noise(l_res):
	for res in l_res:
		noised_img = add_gauss_noise(res.get_original_image(), res.get_var())
		res.set_noised_image(noised_img, res.get_var())


def denoise(l_res):
	print("DENOIZE:")
	total_res = len(l_res)
	one_percent = total_res / 10
	i = 0
	for res in l_res:
		noised_img = res.get_noised_image()
		alg = res.get_alg()
		denoised_img = alg.denoise(noised_img)
		res.set_denoised_image(denoised_img)
		print(i)
		i = i + 1

def plot_by_one_image(store, path, l_alg, l_var, save=True, show=False):
	title = "PSNR_" + path.split("/")[1].split('-')[0]
	print("TITLE: " + title)
	plt.plot()

	plt.title(title)
	plt.xlabel("Variance of gaussian noise")
	plt.ylabel("PSNR")
	plt.grid(True)

	for den in l_alg:
		alg_store = Storage(store.get_results_by_alg(den))

		print("This alg procees  image")
		print(len(alg_store.l_res))
		l_psnr = list()

		for var in l_var:
			l_res = alg_store.get_results_by_image_and_var(path, var)

			if len(l_res) is not 1:
				print("ERROR: some image with same path and same variance")


			res = l_res[0]
			l_psnr.append(res.get_psnr())

		plt.plot(l_var, l_psnr, label=den.get_name())

	plt.legend()

	if show:
		plt.show()

	if save:
		format = "png"
		name_plot = title + "." + format
		plt.savefig(title, fmt=format)



def count_psnr(l_res):
	for res in l_res:
		psnr = PSNR(res.get_original_image(), res.get_denoised_image())
		res.set_psnr(psnr)
		res.save_result()


def chois_optimal_param_billateral():
	l_den = list()
	print("Init vat")
	l_var = init_var(b=0.05, e=0.1)
	print("I have " + str(len(l_var)) + " var")

	########init##########
	print("Init params")
	i = 0
	params = {}
	si = 10
	ss = 1
	params["sigma_i"] = si
	params["sigma_s"] = ss
	params["diameter"] = 2 * (2 * ss + 1)
	i = i + 1
	print("This is " + str(i) + " alg")

	l_den.append(Bilateral(params.copy()))
	'''
	for ss in range(1,5, 1):
		for si in range(1, 10, 3):
			params = {}

			params["sigma_i"] = si
			params["sigma_s"] = ss
			params["diameter"] = 2 * (2*ss+1)
			i = i + 1
			print("This is " + str(i) + " alg")

			l_den.append(Bilateral(params.copy()))
			'''
	###############################
	print("Init image")
	l_im = init_images()
	##############################
	l_res = init_result(l_var, l_im, l_den)

	add_noise(l_res)
	denoise(l_res)
	count_psnr(l_res)

	for im in l_im:
		params = get_opt(im, l_var[0])
		print("for " + im.path + " next param is optimal")
		print(params)

	store = Storage(l_res)
	return l_den
# For each combination alg and image create CLASS result

def get_opt(im, var):
	optimal = "optimal"
	bill_dir = "Bilateral"
	var_dir = str(var)
	json_dir = "json"
	full_path = join(join(join(optimal,bill_dir), var_dir), json_dir)
	jsons = os.listdir(full_path)

	params = {}
	max = 0;
	for js in jsons:
		fp = open(join(full_path, js),'w')
		_dict = json.load(fp)
		if _dict['psnr'] > max:
			params = _dict.copy()
			max = _dict['psnr']

	return params.copy()

def test_all_algorythms():
	l_var = init_var(b=0.1, e=0.2)

	l_im = init_image()
	l_den = init_denoiser()
	l_res = init_result(l_var, l_im, l_den)

	add_noise(l_res)
	denoise(l_res)
	count_psnr(l_res)

	store = Storage(l_res)
	plot_by_one_image(store, l_im[0].path, l_den, l_var=l_var)



def main():
	chois_optimal_param_billateral()
	#test_all_algorythms()

if __name__ == "__main__":
	sys.exit(main())
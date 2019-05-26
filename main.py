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
from common import *

from denoise_algorithms.Bilateral import *
from denoise_algorithms.TVL1 import *
from denoise_algorithms.NL_mean import *
from denoise_algorithms.BM3D import *
from denoise_algorithms.MF import *
from denoise_algorithms.Guided import *

#Init all algorithms of denoise
def init_denoiser():
	lDenoiser = list()

	b = Bilateral()# WORK
	#lDenoiser.append(b)

	tv = TVL1()# WORK
	#lDenoiser.append(tv)

	nl = NL_mean()# WORK
	#lDenoiser.append(nl)

	mf = MF()# NOT WORK ?????
	lDenoiser.append(mf)

	g = Guided()# NOT WORK
	#lDenoiser.append(g)

	bm3d = BM3D()
	#lDenoiser.append(bm3d)

	return lDenoiser

def main():
	optimal = False
	test = True # for test with small gray image
	fold_with_im = ""

	if test:
		fold_with_im = "test_im"
	else:
		fold_with_im = "images"

	l_den = init_denoiser()

	common = Common()
	l_im = common.get_images(fold_with_im)
	l_var = common.get_list_variance(bv=0.005, ev=0.006, sv=0.1)

	common.init_result(l_var, l_im, l_den)
	common.add_noise_to_list()

	lRes = common.get_result()

	for res in lRes:
		nIm = res.get_noised_image()
		alg = res.get_alg()

		print("Denoising" + alg.get_name())
		dIm = alg.denoise(nIm)
		print(alg)
		res.set_denoised_image(dIm)

		psnr = common.PSNR(res.get_original_image(), dIm)
		ssim = common.SSIM(res.get_original_image(), dIm)
		res.set_psnr(psnr)
		res.set_ssim(ssim)

		res.save_result()

	#############optimal param#########################
	if optimal is True:
		for alg in l_den:
			for var in l_var:
				for im in l_im:
					opt_psnr = common.get_optimal_param(alg, var, "psnr", im.get_path())
					opt_ssim = common.get_optimal_param(alg, var, "ssim", im.get_path())



if __name__ == "__main__":
	sys.exit(main())
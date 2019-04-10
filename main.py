import numpy
import sys
import cv2
import os
import fnmatch
import math
import random

from denoise_algorithms.Bilateral import *
from denoise_algorithms.TVL1 import *
from denoise_algorithms.NL_mean import *
from denoise_algorithms.BM3D import *
from denoise_algorithms.MF import *
from denoise_algorithms.Guided import *

class DataImage:
	im = None
	im_n = None # image with noise
	name = "" # name of image
	err_b = -1
	def __init__(self, im, name):
		self.im = im
		self.name = name

	def __generate_error(self, arr, err_b):
		val = arr.copy()

		(w,h,c)= val.shape
		print("i:%d j:%d k:%d" % (w, h, c))
		for i in range(w):
			for j in range(h):
				for k in range(c):
					var = 0
					for n in range(8):
						if random.uniform(0, 1) < err_b:
							var += 1
						var = var << 1

					val[i, j, k] = val[i, j, k] ^ var

		return val

	def add_noise(self, snr=20, pastushok=True):
		img = self.im
		if pastushok:
			row, col, ch = img.shape
			self.err_b = 0.01

			#cv2.imshow("ddd", self.__generate_error(img, 0.001))
			self.im_n =self.__generate_error(img, self.err_b)
			return self.im_n
		else:
			row, col, ch = img.shape
			mean = 0
			var = 1.2
			sigma = var ** snr

			gauss = numpy.random.normal(mean, sigma, (row, col, ch))
			gauss = gauss.reshape(row, col, ch)

			noisy = numpy.clip(img + gauss, a_min=0, a_max=255)
			self.im_n = noisy
			return noisy

def load_image(path_image):
	im = cv2.imread(path_image)
	return im

def load_images(dir, count=-1):
	files = os.listdir(dir)
	mask = "*jpg"
	ims = list()
	for file in files:
		if count is -1 or (fnmatch.fnmatch(file, mask) and count > len(ims)):
			ims.append(DataImage(load_image(dir + "/" + file), file))

	return ims


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


#additive noise
def add_noise(img, snr=20):

	noise = numpy.random.normal(0, numpy.sqrt(snr), img.shape)
	noisy_arr = img + noise

	noisy = numpy.clip(noisy_arr,a_min=0,a_max=255)
	return noisy

#list of Image
def denoise_images(lImages, lDenoiser):
	lImgDen = list()
	for img in lImages:
		for den in lDenoiser:
			den.get_name()
			cv2.imwrite("denoised/WTF.jpg", img.im_n)
			img_d = den.denoise(img)
			cv2.imwrite("denoised/"+den.get_name() + ".jpg", img_d)

			lImgDen.append(img_d)

	return lImgDen

#Init algorithms of denoise
def init_denoiser():
	lDenoiser = list()

	b = Bilateral()
	#lDenoiser.append(b)

	tv = TVL1()
	#lDenoiser.append(tv)

	nl = NL_mean()
	#lDenoiser.append(nl)

	mf = MF()
	lDenoiser.append(mf)

	g = Guided()
	#lDenoiser.append(g)


	bm3d = BM3D()
	#lDenoiser.append(bm3d)

	return lDenoiser

#images - list of Images
def add_noise_to_list(images, snr=20):

	for image in images:
		image.add_noise()

def main():
	snr = 3				# SNR for additive noise
	count_images = 1	#Number of images for denoising
	list_of_denoiser = init_denoiser()

	print("LOAD %s IMAGES" % count_images)
	images = load_images("images", count_images)

	print("ADD NOISE")
	add_noise_to_list(images)

	print("DENOSIE")
	denoise_im = denoise_images(images, list_of_denoiser)




if __name__ == "__main__":
	sys.exit(main())
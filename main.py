import numpy
import sys
import cv2
import os
import fnmatch
import math

from denoise_algorithms.Bilateral import *
from denoise_algorithms.TVL1 import *
from denoise_algorithms.NL_mean import *

def load_image(path_image):
	im = cv2.imread(path_image)
	return im

def load_images(dir, count=-1):
	files = os.listdir(dir)
	mask = "*jpg"
	ims = list()
	for file in files:
		if count is -1 or (fnmatch.fnmatch(file, mask) and count > len(ims)):
			ims.append(load_image(dir+"/"+file))

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

	row, col, ch = img.shape
	mean = 0
	var = 1.2
	sigma = var ** snr

	gauss = numpy.random.normal(mean, sigma, (row, col, ch))
	gauss = gauss.reshape(row, col, ch)



	noisy = numpy.clip(img + gauss,a_min=0,a_max=255)
	#noisy = cv2.randn(img.copy(),0.4,0.5)

	return noisy

def denoise_images(list_of_images, list_of_denoiser):
	img_den = list()
	for img in list_of_images:
		for den in list_of_denoiser:
			den.get_name()
			img_den.append(den.denoise(img))

	return img_den

#Init algorithms of denoise
def init_denoiser():
	list_of_denoiser = list()

	b = Bilateral()
	list_of_denoiser.append(b)

	tv = TVL1()
	list_of_denoiser.append(tv)

	return list_of_denoiser

def main():
	snr = 3				# SNR for additive noise
	count_images = 1	#Number of images for denoising
	list_of_denoiser = init_denoiser()

	print("LOAD IMAGE")
	images = load_images("images", 1)

	print("ADD NOISE")
	noises_im = list()
	noises_im.append(add_noise(images[0]))

	cv2.imwrite("noise.jpg", add_noise(images[0]))
	print("DENOSIE")
	denoise_im = denoise_images(noises_im, list_of_denoiser)

	print("PSNR")
	cv2.imwrite("denoise.jpg", denoise_im[0])
	print(PSNR(images[0], denoise_im[0]))

if __name__ == "__main__":
	sys.exit(main())
import cv2
from  skimage.util import  random_noise
import matplotlib.pyplot as plt
import numpy
from denoise_algorithms.Bilateral import *

def read_image(name):
	return cv2.imread(name)

def add_noise(image, type="g", var=0.005):
	if type == "g":
		return random_noise(image, mode='gaussian', var=var)
	elif(type == "p"):
		return random_noise(image, mode='poisson')
	else:
		return random_noise(image, mode='s&p')

KERNEL_SIZE = 11
def gaus(image, kernel):
	return cv2.GaussianBlur(image, ksize=(kernel,kernel), sigmaX=0)

def box(img):
	return cv2.boxFilter(img,ksize=(KERNEL_SIZE,KERNEL_SIZE), ddepth=-1)

def median(img):
	return cv2.medianBlur(img, ksize=KERNEL_SIZE)

def save_iamge(img, name):
	cv2.imwrite(name, img)


sigmas = [0.01, 0.05, 0.15]
original = read_image("C:/Users/nikit/PycharmProjects/diploma/nikita.jpg")

imgs_n = list()
name_imgs = list()
for sigma in sigmas:
	imgs = numpy.clip(add_noise(original, type='g', var=sigma) * 255, 0, 255).astype("uint8")
	imgs_n.append(imgs)

	name_file = "nikita" + str(sigma)
	name_imgs.append(name_file)

	save_iamge(imgs, name_file + ".png")

radiuses = [3, 5, 7 , 11, 13 , 15, 17, 21]

for radius in radiuses:
	i = 0
	params = {}
	for img in imgs_n:
		img_d = gaus(img, radius)
		#params["diameter"] = radius
		#params["sigma_i"] = 5
		#params["sigma_s"] = 5

		#b = Bilateral(params)
		#img_d = b.denoise(img)


		save_iamge(img_d, str(radius) + name_imgs[i] + ".png")
		i = i + 1




# image = numpy.clip(read_image("C:/Users/nikit/PycharmProjects/diploma/test_im/lena512.png") - 255,0,255)
# image = image[:,:,0]
#
# img_g = numpy.clip(add_noise(image, type='g') * 255,0,255).astype("uint8")
# save_iamge(numpy.clip(add_noise(image, type='p') * 255,0,255).astype("uint8"),"imgpppp.png")
# save_iamge(img_g, "img.png")
# save_iamge(median(img_g), "imgmedian.png")
# save_iamge(box(img_g), "imgbox.png")
# save_iamge(gaus(img_g), "imggaus.png")
#
#
# img_s = numpy.clip(add_noise(image, type='s') * 255,0,255).astype("uint8")
#
# save_iamge(img_s, "simgs.png")
# save_iamge(median(img_s), "simgmedian.png")
# save_iamge(box(img_s), "simgbox.png")
# save_iamge(gaus(img_s), "simggaus.png")



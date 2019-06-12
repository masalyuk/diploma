import cv2
import numpy
from common import *
from  skimage.util import  random_noise
from skimage.restoration  import  denoise_nl_means
from denoise_algorithms.Bilateral import *
PATH = "C:\\Users\\nikit\\PycharmProjects\\diploma\\doc\\img\\"
orig_image = cv2.imread(PATH + "orig.jpg")
noised_image = cv2.imread(PATH + "tv2.png")


common = Common()
print(common.PSNR(orig_image, noised_image))
#27.93
#24.95
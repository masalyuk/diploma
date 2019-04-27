#https://github.com/pfchai/GuidedFilter # for check implementation
import numpy
import cv2
from cv2.ximgproc import guidedFilter
from denoise_algorithms.ADenoiser import *

class Guided(ADenoiser):
    def __init__(self, params=None):
        if params is None:
            self.params = {}
            self.params["radius"] = 5
            self.params["eps"] = 1

        ADenoiser.__init__(self, "Guided", params)


def get_name(self):
    return ADenoiser.get_name(self)


# image - class image
def denoise(self, dataImage):
    image = dataImage.copy()
    denoised_im = numpy.zeros_like(image)
    guidedFilter(image, denoised_im, self.radius, self.eps, dst=denoised_im)
    return denoised_im


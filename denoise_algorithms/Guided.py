#https://github.com/pfchai/GuidedFilter # for check implementation
import numpy
import cv2
from cv2.ximgproc import guidedFilter
from denoise_algorithms.ADenoiser import *

class Guided(ADenoiser):
    def __init__(self, params=None):
        if params is None:
            self.params = {}
            self.params["radius"] = 3
            self.params["eps"] = 1e-6
        else:
            self.params = params

        ADenoiser.__init__(self, "Guided", self.params)


    def get_name(self):
        return ADenoiser.get_name(self)


    # image - class image
    def denoise(self, dataImage):

        image = dataImage.copy().astype("uint8")
        denoised_im = numpy.zeros_like(image)
        denoised_im = guidedFilter(image, image, self.params["radius"], self.params["eps"], dst=denoised_im)

        return denoised_im


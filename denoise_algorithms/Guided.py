#https://github.com/pfchai/GuidedFilter # for check implementation
import numpy
import cv2
from cv2.ximgproc import guidedFilter
from denoise_algorithms.ADenoiser import *

class Guided(ADenoiser):
    def __init__(self, params=None):
        if params is None:
            self.radius = 5
            self.eps = 1
            ADenoiser.__init__(self, "Guided")
        else:
            ADenoiser.__init__(self, "Guided", params)

def __str__(self):
    info = "name: " + str(self.name)
    rad_str = "radius: " + str(self.radius)
    eps_str = "epsilon: " + str(self.eps)

    return info + "\n" + rad_str + "\n" + eps_str


def get_name(self):
    return ADenoiser.get_name(self)


# image - class image
def denoise(self, dataImage):
    (image, name) = self.get_img_name(dataImage)
    if self.params is not None:
        self.radius = self.params.radius
        self.eps = self.params.eps

    denoised_im = numpy.zeros_like(image)

    cv2.imwrite("suma.jpg", image)
    guidedFilter(image, denoised_im, self.radius, self.eps, dst=denoised_im)

    if self.dImgs.get(name) is None:
        self.dImgs[name] = list()

    self.dImgs[name].append(Pair(denoised_im, self.params))

    return denoised_im


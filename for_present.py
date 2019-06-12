import math
from denoise_algorithms.NL_mean import *
from common import *
from  skimage.util import  random_noise
import matplotlib.pyplot as plt
import cv2

path_to_image = "C:\\Users\\nikit\\PycharmProjects\\diploma\\images\\person\\3174790355_2d71cc35bd.jpg"
def fix_diam_sigma_i():
    l_Bilateral=[]
    ps  = range(1,15,1)
    params = {}

    t_s = 7
    s_s = 11
    sim = 35
    for t_s in ps:
        params["template_size"] = t_s
        params["search_size"] = s_s
        params["similar"] = sim
        l_Bilateral.append(NL_mean(params.copy()))

    return (l_Bilateral, ps)

psnrs=[]
params=[]

common = Common()

orig_image = cv2.imread(path_to_image)
noised_image = common.add_noise(orig_image, 0.05)

(lBil, params) = fix_diam_sigma_i()
i = 0
for bil in lBil:
    denoised = bil.denoise(noised_image)

    cv2.imwrite("bil"+str(i+1)+".png", denoised)
    i = i + 1
    psnr = common.PSNR(denoised, orig_image)

    psnrs.append(psnr)



#sort
plt.plot(params,psnrs)
plt.show()
import cv2

path_to_noised_image = "D:/result/person/3526532800_baf1cb40ff.jpg/0.01/noised/3526532800_baf1cb40ff.jpg"
noised_image = cv2.imread(path_to_noised_image)

boxR = [11,13,21]

for r in boxR:
	denoised = cv2.medianBlur(noised_image, ksize=r)
	#denoised = cv2.GaussianBlur(noised_image,ksize=(3*r,3*r) ,sigmaX=r)
	#denoised = cv2.boxFilter(noised_image,ddepth=-1,ksize=(r,r))
	cv2.imwrite("median"+str(r)+".png", denoised)
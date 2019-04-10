import numpy
import cv2
from denoise_algorithms.ADenoiser import *
# this code from https://github.com/wuqiyao20160118/
# Parameters initialization
class BM3D(ADenoiser):
	def __init__(self,params=None):
		if params is None:
			self.sigma = 25

			self.Threshold_Hard3D = 2.7 * self.sigma  # Threshold for Hard Thresholding
			self.First_Match_threshold = 2500  # 用于计算block之间相似度的阈值
			self.Step1_max_matched_cnt = 16  # 组最大匹配的块数
			self.Step1_Blk_Size = 8  # block_Size即块的大小，8*8
			self.Step1_Blk_Step = 3  # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
			self.Step1_Search_Step = 3  # 块的搜索step
			self.Step1_Search_Window = 39  # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

			self.Second_Match_threshold = 400  # 用于计算block之间相似度的阈值
			self.Step2_max_matched_cnt = 32
			self.Step2_Blk_Size = 8
			self.Step2_Blk_Step = 3
			self.Step2_Search_Step = 3
			self.Step2_Search_Window = 39
			#Beta_Kaiser = 2.0
			ADenoiser.__init__(self, "BM3D")
		else:
			ADenoiser.__init__(self,"BM3D", params)

	def __str__(self):
		string = ""
		info 				= "name: "		+ str(self.name)
		string+= ("sigma: "+str(self.sigma) + '\n')
		string+= ("Threshold_Hard3D: "+str(self.Threshold_Hard3D) + '\n')
		string+= ("First_Match_threshold: "+str(self.First_Match_threshold) + '\n')
		string+= ("Step1_max_matched_cnt: "+str(self.Step1_max_matched_cnt) + '\n')
		string+= ("Step1_Blk_Size: "+str(self.Step1_Blk_Size) + '\n')
		string+= ("Step1_Blk_Step: "+str(self.Step1_Blk_Step) + '\n')
		string+= ("Step1_Search_Step: "+str(self.Step1_Search_Step) + '\n')
		string+= ("Step1_Search_Window: "+str(self.Step1_Search_Window) + '\n')
		string+= ("Second_Match_threshold: "+str(self.Second_Match_threshold) + '\n')
		string+= ("Step2_max_matched_cnt: "+str(self.Step2_max_matched_cnt) + '\n')
		string+= ("Step2_Blk_Size: "+str(self.Step2_Blk_Size) + '\n')
		string+= ("Step2_Blk_Step: "+str(self.Step2_Blk_Step) + '\n')
		string+= ("Step2_Search_Step: "+str(self.Step2_Search_Step) + '\n')
		string+= ("Step2_Search_Window: "+str(self.Step2_Search_Window))

		return string
	def get_name(self):
		return ADenoiser.get_name(self)

	def __Locate_blk(self,i, j, blk_step, block_size, width, height):
		if i * blk_step + block_size < width:
			point_x = i * blk_step
		else:
			point_x = width - block_size
		if j * blk_step + block_size < height:
			point_y = j * blk_step
		else:
			point_y = height - block_size
		blockPoint = numpy.array((point_x, point_y), dtype=int)
		return blockPoint

	def __Define_SearchWindow(self,img, BlockPoint, WindowSize, Blk_Size):
		"""
		:param img: input image
		:param BlockPoint: coordinate of the left-top corner of the block
		:param WindowSize: size of the search window
		:param Blk_Size:
		:return: left-top corner point of the search window
		"""
		point_x = BlockPoint[0]
		point_y = BlockPoint[1]
		# get four corner points
		x_min = point_x + Blk_Size / 2 - WindowSize / 2
		y_min = point_y + Blk_Size / 2 - WindowSize / 2
		x_max = x_min + WindowSize
		y_max = y_min + WindowSize
		# check whether the corner points have out of range
		if x_min < 0:
			x_min = 0
		elif x_max > img.shape[0]:
			x_min = img.shape[0] - WindowSize
		if y_min < 0:
			y_min = 0
		elif y_max > img.shape[0]:
			y_min = img.shape[0] - WindowSize
		return numpy.array((x_min, y_min), dtype=int)

	def __step1_fast_match(self, img, BlockPoint):
		x, y = BlockPoint
		blk_size = self.Step1_Blk_Size
		Search_Step = self.Step1_Search_Step
		Threshold = self.First_Match_threshold
		max_matched = self.Step1_max_matched_cnt
		window_size = self.Step1_Search_Window

		blk_positions = numpy.zeros((max_matched, 2), dtype=int)
		similar_blocks_3d = numpy.zeros((max_matched, blk_size, blk_size), dtype=float)
		image = img[x:x + blk_size, y:y + blk_size]
		dct_img = cv2.dct(image.astype(numpy.float64))
		window_location = self.__Define_SearchWindow(img, BlockPoint, window_size, blk_size)
		blk_num = int((window_size - blk_size) / Search_Step)
		window_x, window_y = window_location

		count = 0
		matched_blk_pos = numpy.zeros((blk_num ** 2, 2), dtype=int)
		matched_distance = numpy.zeros(blk_num ** 2, dtype=float)
		similar_blocks = numpy.zeros((blk_num ** 2, blk_size, blk_size), dtype=float)

		for i in range(blk_num):
			for j in range(blk_num):
				search_img = img[window_x:window_x + blk_size, window_y:window_y + blk_size]
				dct_search_img = cv2.dct(search_img.astype(numpy.float64))
				distance = numpy.linalg.norm((dct_img - dct_search_img)) ** 2 / (blk_size ** 2)
				if 0 < distance < Threshold:
					matched_blk_pos[count] = numpy.array((window_x, window_y))
					matched_distance[count] = distance
					similar_blocks[count] = dct_search_img
					count += 1
				window_y += Search_Step
			window_x += Search_Step
			window_y = window_location[1]
		distance = matched_distance[:count]
		sort_index = distance.argsort()

		if count >= max_matched:
			count = max_matched
		else:
			count += 1  # add the template image

		similar_blocks_3d[0] = dct_img
		blk_positions[0] = numpy.array((x, y))
		for i in range(1, count):
			index = sort_index[i - 1]
			similar_blocks_3d[i] = similar_blocks[index]
			blk_positions[i] = matched_blk_pos[index]
		return similar_blocks_3d, blk_positions, count

	def __step1_3DFiltering(self, similar_blocks):
		nonzero_count = 0
		for i in range(similar_blocks.shape[1]):
			for j in range(similar_blocks.shape[2]):
				harr_img = cv2.dct(similar_blocks[:, i, j].astype(numpy.float64))
				harr_img[numpy.abs(harr_img) < self.Threshold_Hard3D] = 0
				nonzero_count += harr_img.nonzero()[0].size
				similar_blocks[:, i, j] = cv2.idct(harr_img)[0]
		return similar_blocks, nonzero_count

	def __integ_hardthreshold(self, similar_blocks, blk_positions, basic_img, weight_img, nonzero_count, matched_num,
							Kaiser=None):
		blk_shape = similar_blocks.shape
		if nonzero_count < 1:
			nonzero_count = 1
		block_wight = (1. / nonzero_count)  # * Kaiser
		for i in range(matched_num):
			point = blk_positions[i, :]
			temp_img = (1. / nonzero_count) * cv2.idct(similar_blocks[i, :, :])  # * Kaiser
			basic_img[point[0]:point[0] + blk_shape[1], point[1]:point[1] + blk_shape[2]] += temp_img
			weight_img[point[0]:point[0] + blk_shape[1], point[1]:point[1] + blk_shape[2]] += block_wight
		return basic_img, weight_img

	def __BM3D_step_1(self, img):
		width, height = img.shape
		block_size = self.Step1_Blk_Size
		blk_step = self.Step1_Blk_Step
		width_num = int((width - block_size) / blk_step)
		height_num = int((height - block_size) / blk_step)
		filtered_img = numpy.zeros(img.shape, dtype=float)
		filter_weight = numpy.zeros(img.shape, dtype=float)

		# K = numpy.kaiser(block_size, Beta_Kaiser)
		# m_Kaiser = numpy.matmul(K.T, K)  # construct a Kaiser window

		for i in range(width_num + 2):
			for j in range(height_num + 2):
				BlockPoint = self.__Locate_blk(i, j, blk_step, block_size, width, height)
				similar_blocks_3d, blk_positions, count = self.__step1_fast_match(img, BlockPoint)
				similar_blocks, nonzero_count = self.__step1_3DFiltering(similar_blocks_3d)
				filtered_img, filter_weight = self.__integ_hardthreshold(similar_blocks, blk_positions, filtered_img,
																  filter_weight, nonzero_count, count)
		filtered_img /= filter_weight
		filtered_img.astype(numpy.uint8)
		return filtered_img

	def __step2_fast_match(self, basic_img, img, BlockPoint):
		x, y = BlockPoint
		blk_size = self.Step2_Blk_Size
		Threshold = self.Second_Match_threshold
		Search_Step = self.Step2_Search_Step
		max_matched = self.Step2_max_matched_cnt
		window_size = self.Step2_Search_Window

		blk_positions = numpy.zeros((max_matched, 2), dtype=int)
		similar_blocks_3d = numpy.zeros((max_matched, blk_size, blk_size), dtype=float)
		basic_similar_blocks_3d = numpy.zeros((max_matched, blk_size, blk_size), dtype=float)

		basic_image = basic_img[x:x + blk_size, y:y + blk_size]
		basic_dct_img = cv2.dct(basic_image.astype(numpy.float64))
		image = img[x:x + blk_size, y:y + blk_size]
		dct_img = cv2.dct(image.astype(numpy.float64))

		window_location = self.__Define_SearchWindow(img, BlockPoint, window_size, blk_size)
		blk_num = int((window_size - blk_size) / Search_Step)
		window_x, window_y = window_location

		count = 0
		matched_blk_pos = numpy.zeros((blk_num ** 2, 2), dtype=int)
		matched_distance = numpy.zeros(blk_num ** 2, dtype=float)
		similar_blocks = numpy.zeros((blk_num ** 2, blk_size, blk_size), dtype=float)

		for i in range(blk_num):
			for j in range(blk_num):
				search_img = basic_img[window_x:window_x + blk_size, window_y:window_y + blk_size]
				dct_search_img = cv2.dct(search_img.astype(numpy.float64))
				distance = numpy.linalg.norm((dct_img - dct_search_img)) ** 2 / (blk_size ** 2)

				if 0 < distance < Threshold:
					matched_blk_pos[count] = numpy.array((window_x, window_y))
					matched_distance[count] = distance
					similar_blocks[count] = dct_search_img
					count += 1
				window_y += Search_Step
			window_x += Search_Step
			window_y = window_location[1]
		distance = matched_distance[:count]
		sort_index = distance.argsort()

		if count >= max_matched:
			count = max_matched
		else:
			count += 1  # add the template image

		basic_similar_blocks_3d[0] = basic_dct_img
		similar_blocks_3d[0] = dct_img
		blk_positions[0] = numpy.array((x, y))
		for i in range(1, count):
			index = sort_index[i - 1]
			basic_similar_blocks_3d[i] = similar_blocks[index]
			blk_positions[i] = matched_blk_pos[index]

			x, y = blk_positions[i]
			#temp_noisy_img = noised_img[x:x + blk_size, y:y + blk_size]
			temp_noisy_img = img[x:x + blk_size, y:y + blk_size]
			similar_blocks_3d[i] = cv2.dct(temp_noisy_img.astype(numpy.float64))

		return basic_similar_blocks_3d, similar_blocks_3d, blk_positions, count

	def __step2_3DFiltering(self, similar_basic_blocks, similar_blocks):
		img_shape = similar_basic_blocks.shape
		Wiener_weight = numpy.zeros((img_shape[1], img_shape[2]), dtype=float)

		for i in range(img_shape[1]):
			for j in range(img_shape[2]):
				temp_vector = similar_basic_blocks[:, i, j]
				dct_temp = cv2.dct(temp_vector)
				norm2 = numpy.matmul(dct_temp.T, dct_temp)
				filter_weight = norm2 / (norm2 + self.sigma ** 2)
				if filter_weight != 0:
					Wiener_weight[i, j] = 1 / (filter_weight ** 2)
				# Wiener_weight[i, j] = 1 / (filter_weight**2 * self.sigma**2)
				temp_vector = similar_blocks[:, i, j]
				dct_temp = cv2.dct(temp_vector) * filter_weight
				similar_basic_blocks[:, i, j] = cv2.idct(dct_temp)[0]

		return similar_basic_blocks, Wiener_weight

	def __integ_Wiener(self, similar_blocks, Wiener_weight, blk_positions, basic_img, weight_img, matched_num):
		img_shape = similar_blocks.shape
		block_weight = Wiener_weight

		for i in range(matched_num):
			point = blk_positions[i]
			temp_img = block_weight * cv2.idct(similar_blocks[i, :, :])
			basic_img[point[0]:point[0] + img_shape[1], point[1]:point[1] + img_shape[2]] += temp_img
			weight_img[point[0]:point[0] + img_shape[1], point[1]:point[1] + img_shape[2]] += block_weight
		return basic_img, weight_img

	def __BM3D_step_2(self, basic_img, img):
		width, height = img.shape
		block_size = self.Step2_Blk_Size
		blk_step = self.Step2_Blk_Step
		width_num = int((width - block_size) / blk_step)
		height_num = int((height - block_size) / blk_step)
		filtered_img = numpy.zeros(img.shape, dtype=float)
		filter_weight = numpy.zeros(img.shape, dtype=float)

		for i in range(width_num + 2):
			for j in range(height_num + 2):
				BlockPoint = self.__Locate_blk(i, j, blk_step, block_size, width, height)
				basic_similar_blocks_3d, similar_blocks_3d, blk_positions, count = self.__step2_fast_match(basic_img, img,
																									BlockPoint)
				similar_basic_blocks, Wiener_weight = self.__step2_3DFiltering(basic_similar_blocks_3d, similar_blocks_3d)
				filtered_img, filter_weight = self.__integ_Wiener(similar_basic_blocks, Wiener_weight, blk_positions,
														   filtered_img,
														   filter_weight, count)
		filtered_img /= filter_weight
		filtered_img.astype(numpy.uint8)

		return filtered_img

	def denoise(self, dataImage):
		(image, name) = self.get_img_name(dataImage)

		denoised_img = numpy.zeros(image.shape)
		final_denoised_img = numpy.zeros(image.shape)

		for ch in range(3):
			denoised_img[:,:,ch] = self.__BM3D_step_1(image[:,:,ch])
			final_denoised_img[:,:,ch] = self.__BM3D_step_2(denoised_img[:,:,ch], image[:,:,ch])

		if self.dImgs.get(name) is None:
			self.dImgs[name] = list()

		self.dImgs[name].append(Pair(final_denoised_img, self.params))
		return final_denoised_img

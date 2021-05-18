import os
import SimpleITK as sitk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import math
import random
import multiprocessing
from tqdm import tqdm
from pipeline import NiftiDataset3D
import threading

def ExtractSliceFromImage(image_input):
	# label = image_input['image']
	# # check if the slice contains label
	# extractor = sitk.ExtractImageFilter()
	# size = [label.GetSize()[0],label.GetSize()[1],0]
	# index = [0,0,i]
	# extractor.SetSize(size)
	# extractor.SetIndex(index)
	# label_ = extractor.Execute(label)

	# binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
	# binaryThresholdFilter.SetLowerThreshold(1)
	# binaryThresholdFilter.SetUpperThreshold(255)
	# binaryThresholdFilter.SetInsideValue(1)
	# binaryThresholdFilter.SetOutsideValue(0)
	# label_ = binaryThresholdFilter.Execute(label_)

	# statFilter = sitk.StatisticsImageFilter()
	# statFilter.Execute(label_)

	# if statFilter.GetSum() > 1:
	# 	slices_list.append([case,i])

	return


class NiftiDataset(object):
	"""
	load image-label pair for training, testing and inference.
	Currently only support linear interpolation method
	Args:
		data_dir (string): Path to data directory.

	image_filename (string): Filename of image data.
	label_filename (string): Filename of label data.
	transforms (list): List of SimpleITK image transformations.
	train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
	"""

	def __init__(self,
		data_dir = '',
		image_filenames = '',
		label_filename = '',
		transforms3D=None,
		transforms2D=None,
		train=False,
		labels=[0,1],
		min_pixel=5,
		drop_ratio=0.1):

		# Init membership variables
		self.data_dir = data_dir
		self.image_filenames = image_filenames
		self.label_filename = label_filename
		self.transforms3D = transforms3D
		self.transforms2D = transforms2D
		self.train = train
		self.labels = labels
		self.min_pixel = min_pixel
		self.drop_ratio = drop_ratio

	def read_image(self,path):
		reader = sitk.ImageFileReader()
		reader.SetFileName(path)
		return reader.Execute()

	def get_dataset(self):
		slices_list = []

		# read all images to generate the candidate slice list
		pbar = tqdm(os.listdir(self.data_dir))

		ignore_files = [
			".DS_Store",
			"@eaDir"
		]

		for case in pbar:
			if case in ignore_files:
				continue

			pbar.set_description("Loading {}...".format(case))

			label = self.read_image(os.path.join(self.data_dir,case,self.label_filename))

			for i in range(label.GetSize()[2]):
				# check if the slice contains label
				extractor = sitk.ExtractImageFilter()
				size = [label.GetSize()[0],label.GetSize()[1],0]
				index = [0,0,i]
				extractor.SetSize(size)
				extractor.SetIndex(index)
				label_ = extractor.Execute(label)

				binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
				binaryThresholdFilter.SetLowerThreshold(1)
				binaryThresholdFilter.SetUpperThreshold(255)
				binaryThresholdFilter.SetInsideValue(1)
				binaryThresholdFilter.SetOutsideValue(0)
				label_ = binaryThresholdFilter.Execute(label_)

				statFilter = sitk.StatisticsImageFilter()
				statFilter.Execute(label_)

				if statFilter.GetSum() > self.min_pixel:
					slices_list.append([case,i])
				elif self.drop(self.drop_ratio):
					slices_list.append([case,i])
				else:
					continue

		# randomize the slices
		random.shuffle(slices_list)

		slices_list_1 = []
		slices_list_2 = []

		for value in slices_list:
			slices_list_1.append(value[0])
			slices_list_2.append(value[1])

		dataset = tf.data.Dataset.from_tensor_slices((slices_list_1,slices_list_2))
		dataset = dataset.map(lambda case, slice_num: tuple(tf.py_func(
			self.input_parser, [case, slice_num], [tf.float32,tf.int32])),
			num_parallel_calls=1)
			# num_parallel_calls=multiprocessing.cpu_count())

		# case_list = os.listdir(self.data_dir)

		# dataset = tf.data.Dataset.from_tensor_slices(case_list)
		# dataset = dataset.map(lambda case: tuple(tf.py_func(
		# 	self.input_parser, [case], [tf.float32,tf.int32])),
		# 	num_parallel_calls=multiprocessing.cpu_count())

		self.dataset = dataset
		self.data_size = len(slices_list)
		return self.dataset

	def drop(self,probability):
		return random.random() <= probability

	# def input_parser(self, case):
	# 	# read image and select the desire slice
	# 	case = case.decode("utf-8")

	# 	image_paths = []
	# 	for channel in range(len(self.image_filenames)):
	# 		image_paths.append(os.path.join(self.data_dir,case,self.image_filenames[channel]))

	# 	# read images
	# 	images = []
	# 	for channel in range(len(image_paths)):
	# 		images.append(self.read_image(image_paths[channel]))

	# 	# cast image
	# 	castImageFilter = sitk.CastImageFilter()
	# 	castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
	# 	for channel in range(len(images)):
	# 		images[channel] = castImageFilter.Execute(images[channel])
	# 		# check header consistency
	# 		sameSize = images[channel].GetSize() == images[0].GetSize()
	# 		sameSpacing = images[channel].GetSpacing() == images[0].GetSpacing()
	# 		sameDirection = images[channel].GetDirection() == images[0].GetDirection()

	# 		if sameSize and sameSpacing and sameDirection:
	# 			continue
	# 		else:
	# 			raise Exception('Header info inconsistent: {}'.format(source_paths[channel]))
	# 			exit()

	# 	label = sitk.Image(images[0].GetSize(), sitk.sitkUInt8)
	# 	label.SetOrigin(images[0].GetOrigin())
	# 	label.SetSpacing(images[0].GetSpacing())
	# 	label.SetDirection(images[0].GetDirection())

	# 	if self.train:
	# 		label_ = self.read_image(os.path.join(self.data_dir, case, self.label_filename))

	# 		# check header consistency
	# 		sameSize = label_.GetSize() == images[0].GetSize()
	# 		sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
	# 		sameDirection = label_.GetDirection() == images[0].GetDirection()
	# 		if not (sameSize and sameSpacing and sameDirection):
	# 			raise Exception('Header info inconsistent: {}'.format(os.path.join(self.data_dir,case, self.label_filename)))
	# 			exit()

	# 		thresholdFilter = sitk.BinaryThresholdImageFilter()
	# 		thresholdFilter.SetOutsideValue(0)
	# 		thresholdFilter.SetInsideValue(1)

	# 		castImageFilter = sitk.CastImageFilter()
	# 		castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
	# 		for channel in range(len(self.labels)):
	# 			thresholdFilter.SetLowerThreshold(self.labels[channel])
	# 			thresholdFilter.SetUpperThreshold(self.labels[channel])
	# 			one_hot_label_image = thresholdFilter.Execute(label_)
	# 			multiFilter = sitk.MultiplyImageFilter()
	# 			one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
	# 			# cast one_hot_label to sitkUInt8
	# 			one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
	# 			one_hot_label_image.SetSpacing(images[0].GetSpacing())
	# 			one_hot_label_image.SetDirection(images[0].GetDirection())
	# 			one_hot_label_image.SetOrigin(images[0].GetOrigin())
	# 			addFilter = sitk.AddImageFilter()
	# 			label = addFilter.Execute(label,one_hot_label_image)

	# 	sample = {'image':images, 'label':label}

	# 	if self.transforms3D:
	# 		for transform in self.transforms3D:
	# 			try:
	# 				sample =transform(sample)
	# 			except:
	# 				print("Dataset preprocessing error: {}".format(os.path.dirname(image_paths[0])))
	# 				exit()

	# 	# extract the desire slice
	# 	images = sample['image']
	# 	label = sample['label']

	# 	# check if the slice contains label
	# 	contain_label = False

	# 	extractor = sitk.ExtractImageFilter()
	# 	size = [label.GetSize()[0],label.GetSize()[1],0]
	# 	extractor.SetSize(size)
		
	# 	binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
	# 	binaryThresholdFilter.SetLowerThreshold(1)
	# 	binaryThresholdFilter.SetUpperThreshold(255)
	# 	binaryThresholdFilter.SetInsideValue(1)
	# 	binaryThresholdFilter.SetOutsideValue(0)
	# 	label_ = binaryThresholdFilter.Execute(label)

	# 	labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
	# 	labelShapeFilter.Execute(label_)
	# 	bbox = labelShapeFilter.GetBoundingBox(1)

	# 	statFilter = sitk.StatisticsImageFilter()
	# 	statFilter.Execute(label_)
	# 	if statFilter.GetSum() < self.min_pixel:
	# 		contain_label = True
	# 		slice_num = np.random.randint(0,label.GetSize()[2])
	# 		index = [0,0,slice_num]
	# 		extractor.SetIndex(index)

	# 	while not contain_label: 
	# 		slice_num = np.random.randint(bbox[2],bbox[2]+bbox[5])
	# 		index = [0,0,slice_num]
	# 		extractor.SetIndex(index)
	# 		label_ = extractor.Execute(label)
	# 		label_ = binaryThresholdFilter.Execute(label_)
		
	# 		statFilter.Execute(label_)

	# 		# will iterate until a sub volume containing label is extracted
	# 		# pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
	# 		# if statFilter.GetSum()/pixel_count<self.min_ratio:
	# 		if statFilter.GetSum()<self.min_pixel:
	# 			contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
	# 		else:
	# 			contain_label = True

	# 	for channel in range(len(images)):
	# 		images[channel] = extractor.Execute(images[channel])

	# 	label = extractor.Execute(label)

	# 	sample = {'image':images, 'label':label}

	# 	if self.transforms2D:
	# 		for transform in self.transforms2D:
	# 			try:
	# 				sample = transform(sample)
	# 			except:
	# 				print("Dataset preprocessing error: {}".format(os.path.dirname(image_paths[0])))
	# 				exit()

	# 	# convert sample to tf tensors
	# 	for channel in range(len(sample['image'])):
	# 		image_np_ = sitk.GetArrayFromImage(sample['image'][channel])
	# 		image_np_ = np.asarray(image_np_,np.float32)
	# 		if channel == 0:
	# 			image_np = image_np_[:,:,np.newaxis]
	# 		else:
	# 			image_np = np.append(image_np,image_np_[:,:,np.newaxis],axis=-1)

	# 	label_np = sitk.GetArrayFromImage(sample['label'])
	# 	label_np = np.asarray(label_np,np.int32)

	# 	return image_np, label_np

	def input_parser(self, case, slice_num):
		# read image and select the desire slice
		case = case.decode("utf-8")
		slice_num = int(slice_num)

		image_paths = []
		for channel in range(len(self.image_filenames)):
			image_paths.append(os.path.join(self.data_dir,case,self.image_filenames[channel]))

		# read images
		images = []
		for channel in range(len(image_paths)):
			images.append(self.read_image(image_paths[channel]))

		# cast image
		for channel in range(len(images)):
			castImageFilter = sitk.CastImageFilter()
			castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
			images[channel] = castImageFilter.Execute(images[channel])
			# check header consistency
			sameSize = images[channel].GetSize() == images[0].GetSize()
			sameSpacing = images[channel].GetSpacing() == images[0].GetSpacing()
			sameDirection = images[channel].GetDirection() == images[0].GetDirection()

			if sameSize and sameSpacing and sameDirection:
				continue
			else:
				raise Exception('Header info inconsistent: {}\nSame size: {}\nSame spacing: {}\nSame direction: {}'.
					format(source_paths[channel],
						sameSize,
						sameSpacing,
						sameDirection))
				exit()

		label = sitk.Image(images[0].GetSize(), sitk.sitkInt32)
		label.SetOrigin(images[0].GetOrigin())
		label.SetSpacing(images[0].GetSpacing())
		label.SetDirection(images[0].GetDirection())

		if self.train:
			label_ = self.read_image(os.path.join(self.data_dir, case, self.label_filename))

			# check header consistency
			sameSize = label_.GetSize() == images[0].GetSize()
			sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
			sameDirection = label_.GetDirection() == images[0].GetDirection()
			if not (sameSize and sameSpacing and sameDirection):
				raise Exception('Header info inconsistent: {}'.format(os.path.join(self.data_dir,case, self.label_filename)))
				exit()

			for channel in range(len(self.labels)):
				thresholdFilter = sitk.BinaryThresholdImageFilter()
				thresholdFilter.SetOutsideValue(0)
				thresholdFilter.SetInsideValue(1)
				thresholdFilter.SetLowerThreshold(self.labels[channel])
				thresholdFilter.SetUpperThreshold(self.labels[channel])
				one_hot_label_image = thresholdFilter.Execute(label_)
				multiFilter = sitk.MultiplyImageFilter()
				one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
				# cast one_hot_label to sitkInt32
				castImageFilter = sitk.CastImageFilter()
				castImageFilter.SetOutputPixelType(sitk.sitkInt32)
				one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
				one_hot_label_image.SetSpacing(images[0].GetSpacing())
				one_hot_label_image.SetDirection(images[0].GetDirection())
				one_hot_label_image.SetOrigin(images[0].GetOrigin())
				addFilter = sitk.AddImageFilter()
				label = addFilter.Execute(label,one_hot_label_image)

		sample = {'image':images, 'label':label}

		if self.transforms3D:
			for transform in self.transforms3D:
				try:
					# print(case, transform.name)
					sample =transform(sample)
					# print(case, transform.name,"3d transform complete")
				except:
					print("Dataset preprocessing error: {}".format(os.path.dirname(image_paths[0])))
					exit()

		# extract the desire slice
		images = sample['image']
		label = sample['label']

		size = [images[0].GetSize()[0],images[0].GetSize()[1],0]
		index = [0,0,int(slice_num)]
		for channel in range(len(images)):
			extractor = sitk.ExtractImageFilter()
			extractor.SetSize(size)
			extractor.SetIndex(index)
			images[channel] = extractor.Execute(images[channel])

		extractor = sitk.ExtractImageFilter()
		extractor.SetSize(size)
		extractor.SetIndex(index)
		label = extractor.Execute(label)

		sample = {'image':images, 'label':label}

		if self.transforms2D:
			for transform in self.transforms2D:
				try:
					# print(case, transform.name)
					sample = transform(sample)
					# print(case, transform.name,"2d transform complete")
				except:
					print("Dataset preprocessing error: {}".format(os.path.dirname(image_paths[0])))
					exit()

		# convert sample to tf tensors
		for channel in range(len(sample['image'])):
			image_np_ = sitk.GetArrayFromImage(sample['image'][channel])
			image_np_ = np.asarray(image_np_,np.float32)
			if channel == 0:
				image_np = image_np_[:,:,np.newaxis]
			else:
				image_np = np.append(image_np,image_np_[:,:,np.newaxis],axis=-1)

		label_np = sitk.GetArrayFromImage(sample['label'])
		label_np = np.asarray(label_np,np.int32)

		# print(case, "convert sitk image to np array complete")

		return image_np, label_np

class ManualNormalization(object):
	"""
	Normalize an image by mapping intensity with given max and min window level
	"""

	def __init__(self,windowMin, windowMax):
		self.name = 'ManualNormalization'
		assert isinstance(windowMax, (int,float))
		assert isinstance(windowMin, (int,float))
		self.windowMax = float(windowMax)
		self.windowMin = float(windowMin)

	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		
		for channel in range(len(image)):
			intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
			intensityWindowingFilter.SetOutputMaximum(255)
			intensityWindowingFilter.SetOutputMinimum(0)
			intensityWindowingFilter.SetWindowMaximum(self.windowMax);
			intensityWindowingFilter.SetWindowMinimum(self.windowMin);
			image[channel] = intensityWindowingFilter.Execute(image[channel])

		return {'image': image, 'label': label}

class Resample(object):
	"""
	Resample the volume in a sample to a given voxel size

	Args:
		voxel_size (float or tuple): Desired output size.
		If float, output volume is isotropic.
		If tuple, output voxel size is matched with voxel size
		Currently only support linear interpolation method
	"""

	def __init__(self, voxel_size):
		self.name = 'Resample'

		assert isinstance(voxel_size, (int, float, tuple, list))
		if isinstance(voxel_size, float):
			self.voxel_size = (voxel_size, voxel_size)
		else:
			assert len(voxel_size) == 2
			self.voxel_size = voxel_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		for image_channel in range(len(image)):
			old_spacing = image[image_channel].GetSpacing()
			old_size = image[image_channel].GetSize()

			new_spacing = self.voxel_size

			new_size = []
			for i in range(2):
				new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
			new_size = tuple(new_size)

			resampler = sitk.ResampleImageFilter()
			resampler.SetInterpolator(sitk.sitkLinear)
			resampler.SetOutputSpacing(new_spacing)
			resampler.SetSize(new_size)

			# resample on image
			resampler.SetOutputOrigin(image[image_channel].GetOrigin())
			resampler.SetOutputDirection(image[image_channel].GetDirection())
			# print("Resampling image...")
			image[image_channel] = resampler.Execute(image[image_channel])

		# resample on segmentation
		resampler = sitk.ResampleImageFilter()
		resampler.SetInterpolator(sitk.sitkLinear)
		resampler.SetOutputSpacing(new_spacing)
		resampler.SetSize(new_size)
		resampler.SetInterpolator(sitk.sitkNearestNeighbor)
		resampler.SetOutputOrigin(label.GetOrigin())
		resampler.SetOutputDirection(label.GetDirection())
		# print("Resampling segmentation...")
		label = resampler.Execute(label)

		return {'image': image, 'label': label}

class Padding(object):
	"""
	Add padding to the image if size is smaller than patch size

	Args:
		output_size (tuple or int): Desired output size. If int, a cubic volume is formed
	"""

	def __init__(self, output_size):
		self.name = 'Padding'

		assert isinstance(output_size, (int, tuple, list))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

		assert all(i > 0 for i in list(self.output_size))

	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		size_old = image[0].GetSize()

		if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]):
			return sample
		else:
			output_size = list(self.output_size)
			if size_old[0] > self.output_size[0]:
				output_size[0] = size_old[0]
			if size_old[1] > self.output_size[1]:
				output_size[1] = size_old[1]

			output_size = tuple(output_size)

			for image_channel in range(len(image)):
				resampler = sitk.ResampleImageFilter()
				resampler.SetOutputSpacing(image[image_channel].GetSpacing())
				resampler.SetSize(output_size)

				# resample on image
				resampler.SetInterpolator(sitk.sitkLinear)
				resampler.SetOutputOrigin(image[image_channel].GetOrigin())
				resampler.SetOutputDirection(image[image_channel].GetDirection())
				image[image_channel] = resampler.Execute(image[image_channel])

			# resample on label
			resampler = sitk.ResampleImageFilter()
			resampler.SetOutputSpacing(image[image_channel].GetSpacing())
			resampler.SetSize(output_size)
			resampler.SetInterpolator(sitk.sitkNearestNeighbor)
			resampler.SetOutputOrigin(label.GetOrigin())
			resampler.SetOutputDirection(label.GetDirection())

			label = resampler.Execute(label)

			return {'image': image, 'label': label}

class RandomCrop(object):
	"""
	Crop randomly the image in a sample. This is usually used for data augmentation.
	Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
	This transformation only applicable in train mode

	Args:
	output_size (tuple or int): Desired output size. If int, cubic crop is made.
	"""

	def __init__(self, output_size, drop_ratio=0.1, min_pixel=1):
		self.name = 'Random Crop'

		assert isinstance(output_size, (int, tuple, list))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

		assert isinstance(drop_ratio, (int,float))
		if drop_ratio >=0 and drop_ratio<=1:
			self.drop_ratio = drop_ratio
		else:
			raise RuntimeError('Drop ratio should be between 0 and 1')

		assert isinstance(min_pixel, int)
		if min_pixel >=0 :
			self.min_pixel = min_pixel
		else:
			raise RuntimeError('Min label pixel count should be integer larger than 0')

	def __call__(self,sample):
		image, label = sample['image'], sample['label']
		size_old = image[0].GetSize()
		size_new = self.output_size

		contain_label = False

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([size_new[0],size_new[1]])

		# statFilter = sitk.StatisticsImageFilter()
		# statFilter.Execute(label)
		# print(statFilter.GetMaximum(), statFilter.GetSum())

		binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
		binaryThresholdFilter.SetLowerThreshold(1)
		binaryThresholdFilter.SetUpperThreshold(255)
		binaryThresholdFilter.SetInsideValue(1)
		binaryThresholdFilter.SetOutsideValue(0)
		label_ = binaryThresholdFilter.Execute(label)

		# check if the whole slice contain label > minimum pixel
		statFilter = sitk.StatisticsImageFilter()
		statFilter.Execute(label_)
		if statFilter.GetSum() < self.min_pixel:
			contain_label = True

		while not contain_label: 
			# get the start crop coordinate in ijk
			if size_old[0] <= size_new[0]:
				start_i = 0
			else:
				start_i = np.random.randint(0, size_old[0]-size_new[0])

			if size_old[1] <= size_new[1]:
				start_j = 0
			else:
				start_j = np.random.randint(0, size_old[1]-size_new[1])

			roiFilter.SetIndex([start_i,start_j])

			label_crop = roiFilter.Execute(label_)

			statFilter.Execute(label_crop)

			# will iterate until a sub volume containing label is extracted
			# pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
			# if statFilter.GetSum()/pixel_count<self.min_ratio:
			if statFilter.GetSum()<self.min_pixel:
				contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
			else:
				contain_label = True

		for image_channel in range(len(image)):
			image[image_channel] = roiFilter.Execute(image[image_channel])
		label = roiFilter.Execute(label)

		return {'image': image, 'label': label}

	def drop(self,probability):
		return random.random() <= probability
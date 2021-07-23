import os
import SimpleITK as sitk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import math
import random
import multiprocessing

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
		transforms=None,
		train=False,
		attention=False,
		labels=[0,1]):

		# Init membership variables
		self.data_dir = data_dir
		self.image_filenames = image_filenames
		self.label_filename = label_filename
		self.transforms = transforms
		self.train = train
		self.labels=labels

	def get_dataset(self):
		case_list = os.listdir(self.data_dir)
		try:
			case_list.remove(".DS_Store")
			case_list.remove("@eaDir")
		except:
			pass

		dataset = tf.data.Dataset.from_tensor_slices(case_list)
		dataset = dataset.map(lambda case: tuple(tf.py_func(
			self.input_parser, [case], [tf.float32,tf.int32])),
			num_parallel_calls=1)
			# num_parallel_calls=multiprocessing.cpu_count())

		self.dataset = dataset
		self.data_size = len(case_list)
		return self.dataset

	def read_image(self,path):
		reader = sitk.ImageFileReader()
		reader.SetFileName(path)
		return reader.Execute()

	def input_parser(self,case):
		case = case.decode("utf-8")

		image_paths = []
		for channel in range(len(self.image_filenames)):
			image_paths.append(os.path.join(self.data_dir,case,self.image_filenames[channel]))

		# read image and label
		images = []
		for channel in range(len(image_paths)):
			image = self.read_image(image_paths[channel])
			images.append(image)

		for channel in range(len(images)):
			# check header same
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

		label = sitk.Image(images[0].GetSize(),sitk.sitkUInt8)
		label.SetOrigin(images[0].GetOrigin())
		label.SetSpacing(images[0].GetSpacing())
		label.SetDirection(images[0].GetDirection())

		if self.train:
			label_ = self.read_image(os.path.join(self.data_dir,case, self.label_filename))

			# check header same
			sameSize = label_.GetSize() == images[0].GetSize()
			sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
			sameDirection = label_.GetDirection() == images[0].GetDirection()
			if not (sameSize and sameSpacing and sameDirection):
				raise Exception('Header info inconsistent: {}\nSame size: {}\nSame spacing: {}\nSame direction: {}'.
					format(os.path.join(self.data_dir,case, self.label_filename),
						sameSize,
						sameSpacing,
						sameDirection
						))
				exit()

			thresholdFilter = sitk.BinaryThresholdImageFilter()
			thresholdFilter.SetOutsideValue(0)
			thresholdFilter.SetInsideValue(1)

			castImageFilter = sitk.CastImageFilter()
			castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
			for channel in range(len(self.labels)):
				thresholdFilter.SetLowerThreshold(self.labels[channel])
				thresholdFilter.SetUpperThreshold(self.labels[channel])
				one_hot_label_image = thresholdFilter.Execute(label_)
				multiFilter = sitk.MultiplyImageFilter()
				one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
				# cast one_hot_label to sitkUInt8
				one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
				one_hot_label_image.SetSpacing(images[0].GetSpacing())
				one_hot_label_image.SetDirection(images[0].GetDirection())
				one_hot_label_image.SetOrigin(images[0].GetOrigin())
				addFilter = sitk.AddImageFilter()
				label = addFilter.Execute(label,one_hot_label_image)

		sample = {'image':images, 'label':label}

		if self.transforms:
			for transform in self.transforms:
				try:
					sample = transform(sample)
				except:
					print("Dataset preprocessing error: {} transform: {}".format(case,transform.name))
					exit()
				
		# convert sample to tf tensors
		for channel in range(len(sample['image'])):
			image_np_ = sitk.GetArrayFromImage(sample['image'][channel])
			image_np_ = np.asarray(image_np_,np.float32)
			# to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
			image_np_ = np.transpose(image_np_,(2,1,0))
			if channel == 0:
				image_np = image_np_[:,:,:,np.newaxis]
			else:
				image_np = np.append(image_np,image_np_[:,:,:,np.newaxis],axis=-1)

		label_np = sitk.GetArrayFromImage(sample['label'])
		label_np = np.asarray(label_np,np.int32)
		# to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
		label_np = np.transpose(label_np,(2,1,0))

		return image_np, label_np

class Normalization(object):
	"""
	Normalize an image to 0 - 255
	"""

	def __init__(self):
		self.name = 'Normalization'

	def __call__(self, sample):
		# normalizeFilter = sitk.NormalizeImageFilter()
		# image, label = sample['image'], sample['label']
		# image = normalizeFilter.Execute(image)
		resacleFilter = sitk.RescaleIntensityImageFilter()
		resacleFilter.SetOutputMaximum(255)
		resacleFilter.SetOutputMinimum(0)
		image, label = sample['image'], sample['label']
		image = resacleFilter.Execute(image)

		return {'image': image, 'label': label}

class RandomFlip(object):
	"""
	Randomly Flip image by user specified axes
	"""

	def __init__(self, axes):
		self.name = 'Flip'
		assert len(axes)>0 and len(axes)<=3
		self.axes = axes

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		flip = np.random.randint(2, size=1)[0]
		if flip:
			flipFilter = sitk.FlipImageFilter()
			for image_channel in range(len(image)):
				flipFilter.SetFlipAxes(self.axes)
				image[image_channel] = flipFilter.Execute(image[image_channel])
			label = flipFilter.Execute(label)

		return {'image': image, 'label': label}

class StatisticalNormalization(object):
	"""
	Normalize an image by mapping intensity with intensity distribution
	
	Args:
		sigma (float): standard deviation/ percentile of intensity value to be excluded
		pre_norm (bool): perform initial normalize of the input image by setting its mean to zero and variance to one
	"""

	def __init__(self, sigma, pre_norm=False):
		self.name = 'StatisticalNormalization'
		assert isinstance(sigma, float)
		self.sigma = sigma
		self.pre_norm=pre_norm

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		for image_channel in range(len(image)):
			if self.pre_norm:
				normalFilter= sitk.NormalizeImageFilter()
				image[image_channel] = normalFilter.Execute(image[image_channel])

			statisticsFilter = sitk.StatisticsImageFilter()
			statisticsFilter.Execute(image[image_channel])

			intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
			intensityWindowingFilter.SetOutputMaximum(255)
			intensityWindowingFilter.SetOutputMinimum(0)
			intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+self.sigma*statisticsFilter.GetSigma())
			intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-self.sigma*statisticsFilter.GetSigma())

			image[image_channel] = intensityWindowingFilter.Execute(image[image_channel])

		return {'image': image, 'label': label}

class ExtremumNormalization(object):
	"""
	Normalize an image by mapping intensity with intensity maximum and minimum
	"""

	def __init__(self, percent=0.05):
		self.name = 'ExtremumNormalization'
		assert isinstance(percent, float)
		self.percent = percent

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		for image_channel in range(len(image)):
			statisticsFilter = sitk.StatisticsImageFilter()
			statisticsFilter.Execute(image[image_channel])

			intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
			intensityWindowingFilter.SetOutputMaximum(255)
			intensityWindowingFilter.SetOutputMinimum(0)
			windowMax = (statisticsFilter.GetMaximum() - statisticsFilter.GetMinimum())*(1-self.percent) + statisticsFilter.GetMinimum()
			windowMin = (statisticsFilter.GetMaximum() - statisticsFilter.GetMinimum())*self.percent + statisticsFilter.GetMinimum()
			intensityWindowingFilter.SetWindowMaximum(windowMax);
			intensityWindowingFilter.SetWindowMinimum(windowMin);

			image[image_channel] = intensityWindowingFilter.Execute(image[image_channel])

		return {'image': image, 'label': label}

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
		intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
		intensityWindowingFilter.SetOutputMaximum(255)
		intensityWindowingFilter.SetOutputMinimum(0)
		intensityWindowingFilter.SetWindowMaximum(self.windowMax);
		intensityWindowingFilter.SetWindowMinimum(self.windowMin);

		for channel in range(len(image)):
			image[channel] = intensityWindowingFilter.Execute(image[channel])

		return {'image': image, 'label': label}

class Reorient(object):
	"""
	(Beta) Function to orient image in specific axes order
	The elements of the order array must be an permutation of the numbers from 0 to 2.
	"""

	def __init__(self, order):
		self.name = 'Reoreient'
		assert isinstance(order, (int, tuple, list))
		assert len(order) == 3
		self.order = order

	def __call__(self, sample):
		reorientFilter = sitk.PermuteAxesImageFilter()
		reorientFilter.SetOrder(self.order)
		image = reorientFilter.Execute(sample['image'])
		label = reorientFilter.Execute(sample['label'])

		return {'image': image, 'label': label}

class Invert(object):
	"""
	Invert the image intensity from 0-255 
	"""

	def __init__(self):
		self.name = 'Invert'

	def __call__(self, sample):
		invertFilter = sitk.InvertIntensityImageFilter()
		image = invertFilter.Execute(sample['image'],255)
		label = sample['label']

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

		assert isinstance(voxel_size, (float, tuple, list))
		if isinstance(voxel_size, float):
			self.voxel_size = (voxel_size, voxel_size, voxel_size)
		else:
			assert len(voxel_size) == 3
			self.voxel_size = voxel_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		for image_channel in range(len(image)):
			old_spacing = image[image_channel].GetSpacing()
			old_size = image[image_channel].GetSize()

			new_spacing = self.voxel_size

			new_size = []
			for i in range(3):
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
			assert len(output_size) == 3
			self.output_size = output_size

		assert all(i > 0 for i in list(self.output_size))

	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		size_old = image[0].GetSize()

		if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
			return sample
		else:
			output_size = list(self.output_size)
			if size_old[0] > self.output_size[0]:
				output_size[0] = size_old[0]
			if size_old[1] > self.output_size[1]:
				output_size[1] = size_old[1]
			if size_old[2] > self.output_size[2]:
				output_size[2] = size_old[2]

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
			assert len(output_size) == 3
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
		roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

		# statFilter = sitk.StatisticsImageFilter()
		# statFilter.Execute(label)
		# print(statFilter.GetMaximum(), statFilter.GetSum())

		binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
		binaryThresholdFilter.SetLowerThreshold(1)
		binaryThresholdFilter.SetUpperThreshold(255)
		binaryThresholdFilter.SetInsideValue(1)
		binaryThresholdFilter.SetOutsideValue(0)
		label_ = binaryThresholdFilter.Execute(label)

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

			if size_old[2] <= size_new[2]:
				start_k = 0
			else:
				start_k = np.random.randint(0, size_old[2]-size_new[2])

			roiFilter.SetIndex([start_i,start_j,start_k])

			label_crop = roiFilter.Execute(label_)
			statFilter = sitk.StatisticsImageFilter()
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

class RandomNoise(object):
	"""
	Randomly noise to the image in a sample. This is usually used for data augmentation.
	"""
	def __init__(self,sigma=5):
		self.name = 'Random Noise'
		self.sigma = sigma

	def __call__(self, sample):
		self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
		self.noiseFilter.SetMean(0)
		self.noiseFilter.SetStandardDeviation(self.sigma)

		# print("Normalizing image...")
		image, label = sample['image'], sample['label']

		for image_channel in range(len(image)):
			image[image_channel] = self.noiseFilter.Execute(image[image_channel])		

		return {'image': image, 'label': label}

class ConfidenceCrop(object):
	"""
	Crop the image in a sample that is certain distance from individual labels center. 
	This is usually used for data augmentation with very small label volumes.
	The distance offset from connected label centroid is model by Gaussian distribution with mean zero and user input sigma (default to be 2.5)
	i.e. If n isolated labels are found, one of the label's centroid will be randomly selected, and the cropping zone will be offset by following scheme:
	s_i = np.random.normal(mu, sigma*crop_size/2), 1000)
	offset_i = random.choice(s_i)
	where i represents axis direction
	A higher sigma value will provide a higher offset

	Args:
		output_size (tuple or int): Desired output size. If int, cubic crop is made.
		sigma (float): Normalized standard deviation value.
	"""

	def __init__(self, output_size, sigma=2.5):
		self.name = 'Confidence Crop'

		assert isinstance(output_size, (int, tuple, list))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

		assert isinstance(sigma, (float, tuple, list))
		if isinstance(sigma, float) and sigma >= 0:
			self.sigma = (sigma,sigma,sigma)
		else:
			assert len(sigma) == 3
			self.sigma = sigma

	def __call__(self,sample):
		image, label = sample['image'], sample['label']
		size_new = self.output_size

		# guarantee label type to be integer
		castFilter = sitk.CastImageFilter()
		castFilter.SetOutputPixelType(sitk.sitkInt8)
		label = castFilter.Execute(label)

		ccFilter = sitk.ConnectedComponentImageFilter()
		labelCC = ccFilter.Execute(label)

		labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		labelShapeFilter.Execute(labelCC)

		if labelShapeFilter.GetNumberOfLabels() == 0:
			# handle image without label
			selectedLabel = 0
			centroid = (int(self.output_size[0]/2), int(self.output_size[1]/2), int(self.output_size[2]/2))
		else:
			# randomly select of the label's centroid
			selectedLabel = random.randint(1,labelShapeFilter.GetNumberOfLabels())
			centroid = label.TransformPhysicalPointToIndex(labelShapeFilter.GetCentroid(selectedLabel))

		centroid = list(centroid)

		start = [-1,-1,-1] #placeholder for start point array
		end = [self.output_size[0]-1, self.output_size[1]-1,self.output_size[2]-1] #placeholder for start point array
		offset = [-1,-1,-1] #placeholder for start point array
		for i in range(3):
			# edge case
			if centroid[i] < (self.output_size[i]/2):
				centroid[i] = int(self.output_size[i]/2)
			elif (image.GetSize()[i]-centroid[i]) < (self.output_size[i]/2):
				centroid[i] = image.GetSize()[i] - int(self.output_size[i]/2) -1

			# get start point
			while ((start[i]<0) or (end[i]>(image.GetSize()[i]-1))):
				offset[i] = self.NormalOffset(self.output_size[i],self.sigma[i])
				start[i] = centroid[i] + offset[i] - int(self.output_size[i]/2)
				end[i] = start[i] + self.output_size[i] - 1

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize(self.output_size)
		roiFilter.SetIndex(start)
		croppedImage = roiFilter.Execute(image)
		croppedLabel = roiFilter.Execute(label)

		return {'image': croppedImage, 'label': croppedLabel}

	def NormalOffset(self,size, sigma):
		s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
		return int(round(random.choice(s)))

class ConfidenceCrop2(object):
	"""
	Crop the image in a sample that is certain distance from individual labels center. 
	This is usually used for data augmentation with very small label volumes.
	The distance offset from connected label bounding box center is a uniform random distribution within user specified range
	Regions containing label is considered to be positive while regions without label is negative. This distribution is governed by the user defined probability

	Args:
		output_size (tuple or int): Desired output size. If int, cubic crop is made.
		range (int): Bounding box random offset max value
		probability (float): Probability to get positive labels
	"""

	def __init__(self, output_size, rand_range=3,probability=0.5, random_empty_region=False):
		self.name = 'Confidence Crop 2'

		assert isinstance(output_size, (int, tuple, list))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

		assert isinstance(rand_range, (int,tuple, list))
		if isinstance(rand_range, int) and rand_range >= 0:
			self.rand_range = (rand_range,rand_range,rand_range)
		else:
			assert len(rand_range) == 3
			self.rand_range = rand_range

		assert isinstance(probability, float)
		if probability >= 0 and probability <= 1:
			self.probability = probability

		assert isinstance(random_empty_region, bool)
		self.random_empty_region=random_empty_region

	def __call__(self,sample):
		image, label = sample['image'], sample['label']
		size_new = self.output_size

		# guarantee label type to be integer
		castFilter = sitk.CastImageFilter()
		castFilter.SetOutputPixelType(sitk.sitkInt16)
		label = castFilter.Execute(label)

		# choose whether positive or negative label to crop
		zerosList = [0]*int(10*(1-self.probability))
		onesList = [1]*int(10*self.probability)
		choiceList = []
		choiceList.extend(zerosList)
		choiceList.extend(onesList)
		labelType = random.choice(choiceList)

		if labelType == 0:
			# randomly pick a region
			if self.random_empty_region:
				image, label = self.RandomEmptyRegion(image,label)
			else:
				image, label = self.RandomRegion(image,label)
		else:
			# get the number of labels
			ccFilter = sitk.ConnectedComponentImageFilter()
			labelCC = ccFilter.Execute(label)
			labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
			labelShapeFilter.Execute(labelCC)

			if labelShapeFilter.GetNumberOfLabels() == 0:
				if self.random_empty_region:
					image, label = self.RandomEmptyRegion(image,label)
				else:
					image, label = self.RandomRegion(image,label)
			else:
				selectedLabel = random.choice(range(0,labelShapeFilter.GetNumberOfLabels())) + 1 
				selectedBbox = labelShapeFilter.GetBoundingBox(selectedLabel)
				index = [0,0,0]
				for i in range(3):
					index[i] = selectedBbox[i] + int(selectedBbox[i+3]/2) - int(self.output_size[i]/2) + random.choice(range(-1*self.rand_range[i],self.rand_range[i]+1))
					if image[0].GetSize()[i] - index[i] - 1 < self.output_size[i]:
						index[i] = image[0].GetSize()[i] - self.output_size[i] - 1
					if index[i]<0:
						index[i] = 0

				roiFilter = sitk.RegionOfInterestImageFilter()
				roiFilter.SetSize(self.output_size)
				roiFilter.SetIndex(index)

				for image_channel in range(len(image)):
					image[image_channel] = roiFilter.Execute(image[image_channel])
				label = roiFilter.Execute(label)

		return {'image': image, 'label': label}

	def RandomEmptyRegion(self,image, label):
		index = [0,0,0]
		contain_label = False
		while not contain_label:
			for i in range(3):
				if label.GetSize()[i]-self.output_size[i] == 0:
					index[i] = 0
				else:
					index[i] = random.choice(range(0,label.GetSize()[i]-self.output_size[i]-1))
			roiFilter = sitk.RegionOfInterestImageFilter()
			roiFilter.SetSize(self.output_size)
			roiFilter.SetIndex(index)
			label_ = roiFilter.Execute(label)
			statFilter = sitk.StatisticsImageFilter()
			statFilter.Execute(label_)

			if statFilter.GetSum() < 1:
				for image_channel in range(len(image)):
					label = label_
					image[image_channel] = roiFilter.Execute(image[image_channel])
				contain_label = True
				break
		return image,label

	def RandomRegion(self,image, label):
		index = [0,0,0]
		for i in range(3):
			if label.GetSize()[i]-self.output_size[i] == 0:
				index[i] = 0
			else:
				index[i] = random.choice(range(0,label.GetSize()[i]-self.output_size[i]-1))
		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize(self.output_size)
		roiFilter.SetIndex(index)
		label = roiFilter.Execute(label)

		for image_channel in range(len(image)):
			image[image_channel] = roiFilter.Execute(image[image_channel])
			
		return image,label

class BSplineDeformation(object):
	"""
	Image deformation with a sparse set of control points to control a free form deformation.
	Details can be found here: 
	https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf
	https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html

	Args:
		randomness (int,float): BSpline deformation scaling factor, default is 10.
	"""

	def __init__(self, randomness=10):
		self.name = 'BSpline Deformation'

		assert isinstance(randomness, (int,float))
		if randomness > 0:
			self.randomness = randomness
		else:
			raise RuntimeError('Randomness should be non zero values')

	def __call__(self,sample):
		image, label = sample['image'], sample['label']
		spline_order = 3
		domain_physical_dimensions = [image.GetSize()[0]*image.GetSpacing()[0],image.GetSize()[1]*image.GetSpacing()[1],image.GetSize()[2]*image.GetSpacing()[2]]

		bspline = sitk.BSplineTransform(3, spline_order)
		bspline.SetTransformDomainOrigin(image.GetOrigin())
		bspline.SetTransformDomainDirection(image.GetDirection())
		bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
		bspline.SetTransformDomainMeshSize((10,10,10))

		# Random displacement of the control points.
		originalControlPointDisplacements = np.random.random(len(bspline.GetParameters()))*self.randomness
		bspline.SetParameters(originalControlPointDisplacements)

		image = sitk.Resample(image, bspline)
		label = sitk.Resample(label, bspline)
		return {'image': image, 'label': label}

	def NormalOffset(self,size, sigma):
		s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
		return int(round(random.choice(s)))

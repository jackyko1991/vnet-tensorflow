import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import math
import random

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
    image_filename = '',
    label_filename = '',
    transforms=None,
    train=False):

    # Init membership variables
    self.data_dir = data_dir
    self.image_filename = image_filename
    self.label_filename = label_filename
    self.transforms = transforms
    self.train = train

  def get_dataset(self):
    image_paths = []
    label_paths = []
    for case in os.listdir(self.data_dir):
      image_paths.append(os.path.join(self.data_dir,case,self.image_filename))
      label_paths.append(os.path.join(self.data_dir,case,self.label_filename))

    dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))

    dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(
      self.input_parser, [image_path, label_path], [tf.float32,tf.int32])))

    self.dataset = dataset
    self.data_size = len(image_paths)
    return self.dataset

  def read_image(self,path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    return reader.Execute()

  def input_parser(self,image_path, label_path):
    # read image and label
    image = self.read_image(image_path.decode("utf-8"))
    if self.train:
      label = self.read_image(label_path.decode("utf-8"))
    else:
      label = sitk.Image(image.GetSize(),sitk.sitkUInt32)
      label.SetOrigin(image.GetOrigin())
      label.SetSpacing(image.GetSpacing())

    sample = {'image':image, 'label':label}

    if self.transforms:
      for transform in self.transforms:
        sample = transform(sample)

    # convert sample to tf tensors
    image_np = sitk.GetArrayFromImage(sample['image'])
    label_np = sitk.GetArrayFromImage(sample['label'])

    image_np = np.asarray(image_np,np.float32)
    label_np = np.asarray(label_np,np.int32)

    return image_np, label_np

class Normalization(object):
  """
  Normalize an image by setting its mean to zero and variance to one
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

    return {'image':image, 'label':label}

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

    assert isinstance(voxel_size, (float, tuple))
    if isinstance(voxel_size, float):
      self.voxel_size = (voxel_size, voxel_size, voxel_size)
    else:
      assert len(voxel_size) == 3
      self.voxel_size = voxel_size

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    
    new_spacing = self.voxel_size

    new_size = []
    for i in range(3):
      new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
    new_size = tuple(new_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(2)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    # resample on image
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    # print("Resampling image...")
    image = resampler.Execute(image)

    # resample on segmentation
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

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert all(i > 0 for i in list(self.output_size))

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_old = image.GetSize()

    if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
      return sample
    else:
      resampler = sitk.ResampleImageFilter()
      resampler.SetInterpolator(2)
      resampler.SetOutputSpacing(image.GetSpacing())
      resampler.SetSize(self.output_size)

      # resample on image
      resampler.SetOutputOrigin(image.GetOrigin())
      resampler.SetOutputDirection(image.GetDirection())
      image = resampler.Execute(image)

      # resample on label
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

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert isinstance(drop_ratio, float)
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
    size_old = image.GetSize()
    size_new = self.output_size

    contain_label = False

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

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

      label_crop = roiFilter.Execute(label)
      statFilter = sitk.StatisticsImageFilter()
      statFilter.Execute(label_crop)

      # will iterate until a sub volume containing label is extracted
      # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
      # if statFilter.GetSum()/pixel_count<self.min_ratio:
      if statFilter.GetSum()<self.min_pixel:
        contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
      else:
        contain_label = True

    image_crop = roiFilter.Execute(image)

    return {'image': image_crop, 'label': label_crop}

  def drop(self,probability):
    return random.random() <= probability

class RandomNoise(object):
  """
  Randomly noise to the image in a sample. This is usually used for data augmentation.
  """
  def __init__(self):
    self.name = 'Random Noise'

  def __call__(self, sample):
    self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
    self.noiseFilter.SetMean(0)
    self.noiseFilter.SetStandardDeviation(0.1)

    # print("Normalizing image...")
    image, label = sample['image'], sample['label']
    image = self.noiseFilter.Execute(image)

    return {'image': image, 'label': label}

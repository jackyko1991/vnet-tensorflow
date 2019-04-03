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
    image = reader.Execute()
    return image

  def input_parser(self,image_path, label_path):
    # read image and label
    image = self.read_image(image_path.decode("utf-8"))
     # cast image and label
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkInt16)
    image = castImageFilter.Execute(image)

    if self.train:
      label = self.read_image(label_path.decode("utf-8"))
      castImageFilter.SetOutputPixelType(sitk.sitkInt8)
      label = castImageFilter.Execute(label)
    else:
      label = sitk.Image(image.GetSize(),sitk.sitkInt8)
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

    # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
    image_np = np.transpose(image_np,(2,1,0))
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

class StatisticalNormalization(object):
  """
  Normalize an image by mapping intensity with intensity distribution
  """

  def __init__(self, sigma):
    self.name = 'StatisticalNormalization'
    assert isinstance(sigma, float)
    self.sigma = sigma

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    statisticsFilter = sitk.StatisticsImageFilter()
    statisticsFilter.Execute(image)

    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+self.sigma*statisticsFilter.GetSigma());
    intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-self.sigma*statisticsFilter.GetSigma());

    image = intensityWindowingFilter.Execute(image)

    return {'image': image, 'label': label}

class ManualNormalization(object):
  """
  Normalize an image by mapping intensity with given max and min window level
  """

  def __init__(self,windowMin, windowMax):
    self.name = 'ManualNormalization'
    assert isinstance(windowMax, (int,float))
    assert isinstance(windowMin, (int,float))
    self.windowMax = windowMax
    self.windowMin = windowMin

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(self.windowMax);
    intensityWindowingFilter.SetWindowMinimum(self.windowMin);

    image = intensityWindowingFilter.Execute(image)

    return {'image': image, 'label': label}

class Reorient(object):
  """
  (Beta) Function to orient image in specific axes order
  The elements of the order array must be an permutation of the numbers from 0 to 2.
  """

  def __init__(self, order):
    self.name = 'Reoreient'
    assert isinstance(order, (int, tuple))
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
      self.output_size = list(self.output_size)
      if size_old[0] > self.output_size[0]:
        self.output_size[0] = size_old[0]
      if size_old[1] > self.output_size[1]:
        self.output_size[1] = size_old[1]
      if size_old[2] > self.output_size[2]:
        self.output_size[2] = size_old[2]
 
      self.output_size = tuple(self.output_size)

      resampler = sitk.ResampleImageFilter()
      resampler.SetOutputSpacing(image.GetSpacing())
      resampler.SetSize(self.output_size)

      # resample on image
      resampler.SetInterpolator(2)
      resampler.SetOutputOrigin(image.GetOrigin())
      resampler.SetOutputDirection(image.GetDirection())
      image = resampler.Execute(image)

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

    assert isinstance(output_size, (int, tuple))
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
    size_old = image.GetSize()
    size_new = self.output_size

    contain_label = False

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

    # statFilter = sitk.StatisticsImageFilter()
    # statFilter.Execute(label)
    # print(statFilter.GetMaximum(), statFilter.GetSum())

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

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert isinstance(sigma, (float, tuple))
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

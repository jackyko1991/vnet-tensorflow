import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np

# class BatchGenerator:
#   def __init__(self,data_dir,
#     input_batch_shape,
#     output_batch_shape,
#     image_filename = '',
#     label_filename = '',
#     resample=False, 
#     normalization=True,
#     padding=True,
#     randomNoise=False,
#     shuffle=False):

#     # Init membership variables
#     self.data_dir = data_dir
#     self.input_batch_shape = input_batch_shape
#     self.output_batch_shape = output_batch_shape
#     self.image_filename = image_filename
#     self.label_filename = label_filename
#     self.resample = resample
#     self.normalization = normalization
#     self.padding = padding
#     self.randomNoise = randomNoise
#     self.shuffle = shuffle

#     self.pointer = 0

#     # first 4 dimension of input/output batches should have same size
#     if self.input_batch_shape[0:3] != self.output_batch_shape[0:3]:
#       raise ValueError("input output batch dimension does not match")

#     self.scan_data_dir(self.data_dir)

#     if self.shuffle:
#       self.shuffle_data()

#   def scan_data_dir(self,data_dir):
#     """
#     Scan the image and label files to get paths
#     """
#     self.image_paths = []
#     self.label_paths = []
#     for case in os.listdir(data_dir):
#       image_path = os.path.join(data_dir,case,'img.nii.gz')
#       label_path = os.path.join(data_dir,case,'label.nii.gz')
#       # check existence
#       if (os.path.exists(image_path) and os.path.exists(label_path)):
#         self.image_paths.append(image_path)
#         self.label_paths.append(label_path)
#       else:
#         print('image/label in ' + case + ' not exist, ignored.')
    
#     # store total number of data
#     self.data_size = len(self.image_paths)

#   def shuffle_data(self):
#     """
#     Random shuffle the images and labels
#     """

#     image_paths = self.image_paths.copy()
#     label_paths = self.label_paths.copy()

#     self.image_paths = []
#     self.label_paths = []

#     # create the list of permuted index and shuffle data according to list
#     idx = np.random.permutation(len(image_paths))

#     for i in idx:
#       self.image_paths.append(image_paths[i])
#       self.label_paths.append(label_paths[i])
    
#   def next_batch(self):
#     """
#       This function gets the next n ( = batch_size) images and labels from the path list
#       and loads them into memory 
#     """
    
#     # Get next batch of image and label paths
#     image_paths = self.image_paths[self.pointer:self.pointer+self.input_batch_shape[0]]
#     label_paths = self.label_paths[self.pointer:self.pointer+self.input_batch_shape[0]]

#     # update pointer
#     self.pointer += self.input_batch_shape[0]

#     # Read images

class NiftiDataset:
  def __init__(self,
    data_dir = '',
    input_batch_shape = (),
    output_batch_shape = (),
    image_filename = '',
    label_filename = '',
    resample=False, 
    normalization=True,
    padding=True,
    randomNoise=False):

    # Init membership variables
    self.data_dir = data_dir
    self.input_batch_shape = input_batch_shape
    self.output_batch_shape = output_batch_shape
    self.image_filename = image_filename
    self.label_filename = label_filename
    self.resample = resample
    self.normalization = normalization
    self.padding = padding
    self.randomNoise = randomNoise

  def get_dataset(self):
    image_paths = []
    label_paths = []
    for case in os.listdir(self.data_dir):
      image_paths.append(os.path.join(self.data_dir,case,self.image_filename))
      label_paths.append(os.path.join(self.data_dir,case,self.label_filename))

    dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))

    dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(
      self.input_parser, [image_path, label_path], [tf.string,tf.string])))

    self.dataset = dataset

    # print(self.image_filename)
    # exit()
    return self.dataset

  def read_image(self,path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    return reader.Execute()

  def normalize_image(self,image):
    normalizeFilter = sitk.NoralizeImageFilter()
    image = normalizeFilter.Execute(image)
    return image

  def input_parser(self,image_path, label_path):
    # read image and label
    image = self.read_image(image_path.decode("utf-8"))
    label = self.read_image(label_path.decode("utf-8"))

    # image normalization
    if self.normalization:
      image = self.normalize_image(image)

    return image_path, label_path

  # def inputs(data_dir,input_batch_shape,output_batch_shape):
  #   """Construct input for vnet training using the Reader ops.
  #   Args:
  #     data_dir: Path to the data directory.
  #     batch_shape: Shape of the batch.
  #   Returns:
  #     images: Images. 5D tensor of [batch_size, height, width, depth, input_channel] size.
  #     labels: Labels. 5D tensor of [batch_size, height, width, depth, output_channel] size.
  #   """

  #   image_paths = []
  #   label_paths = []
  #   for case in os.listdir(data_dir):
  #     image_paths.append(os.path.join(data_dir,case,'img.nii.gz'))
  #     label_paths.append(os.path.join(data_dir,case,'label.nii.gz'))

  #   dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))

  #   dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(
  #     _input_parser, [image_path, label_path], [tf.float32,tf.int32])))
    
  #   return dataset

# def augumented_inputs(data_dir,input_batch_shape,output_batch_shape):
#   """Construct augumented input for vnet training using the Reader ops.
#   Args:
#     data_dir: Path to the data directory.
#     batch_shape: Shape of the batch.
#   Returns:
#     images: Images. 5D tensor of [batch_size, height, width, depth, input_channel] size.
#     labels: Labels. 5D tensor of [batch_size, height, width, depth, output_channel] size.
#   """
#   inputs(data_dir,input_batch_shape, output_batch_shape)
  

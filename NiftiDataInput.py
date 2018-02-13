import SimpleITK as sitk
import tensorflow as tf

def read_image(path):
  reader = sitk.ImageFileReader()
  reader.SetFilename(path)
  return reader.Execute()

def _parse_function(image_path, label_path):
  print(image_path)
  print(label_path)
  image = read_image(image_path)
  label = read_image(label_path)

def inputs(data_dir,input_batch_shape,output_batch_shape):
  """Construct input for vnet training using the Reader ops.
  Args:
    data_dir: Path to the data directory.
    batch_shape: Shape of the batch.
  Returns:
    images: Images. 5D tensor of [batch_size, height, width, depth, input_channel] size.
    labels: Labels. 5D tensor of [batch_size, height, width, depth, output_channel] size.
  """

  image_paths = []
  label_paths = []
  for case in os.listdir(data_dir):
    image_paths.append(os.path.join(data_dir,case,'image.nii'))
    image_paths.append(os.path.join(data_dir,case,'label.nii'))
  
  image_paths = tf.constant(image_paths)
  label_paths = tf.constant(label_paths)

  dataset = tf.data.Dataset.from_tensor_slices((image_paths,labels))

  dataset = dataset.map(_parse_function)

def augumented_inputs(data_dir,input_batch_shape,output_batch_shape):
    """Construct augumented input for vnet training using the Reader ops.
  Args:
    data_dir: Path to the data directory.
    batch_shape: Shape of the batch.
  Returns:
    images: Images. 5D tensor of [batch_size, height, width, depth, input_channel] size.
    labels: Labels. 5D tensor of [batch_size, height, width, depth, output_channel] size.
  """

  

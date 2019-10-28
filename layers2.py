import tensorflow as tf
import numpy as np

def xavier_initializer_convolution(shape, dist='uniform', lambda_initializer=True):
	"""
		Xavier initializer for N-D convolution patches. input_activations = patch_volume * in_channels;
		output_activations = patch_volume * out_channels; Uniform: lim = sqrt(3/(input_activations + output_activations))
		Normal: stddev =  sqrt(6/(input_activations + output_activations))
		:param shape: The shape of the convolution patch i.e. spatial_shape + [input_channels, output_channels]. The order of
		input_channels and output_channels is irrelevant, hence this can be used to initialize deconvolution parameters.
		:param dist: A string either 'uniform' or 'normal' determining the type of distribution
		:param lambda_initializer: Whether to return the initial actual values of the parameters (True) or placeholders that
		are initialized when the session is initiated
		:return: A numpy araray with the initial values for the parameters in the patch
	"""
	s = len(shape) - 2
	num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
	if dist == 'uniform':
		lim = np.sqrt(6. / num_activations)
		if lambda_initializer:
			return np.random.uniform(-lim, lim, shape).astype(np.float32)
		else:
			return tf.random_uniform(shape, minval=-lim, maxval=lim)
	if dist == 'normal':
		stddev = np.sqrt(3. / num_activations)
		if lambda_initializer:
			return np.random.normal(0, stddev, shape).astype(np.float32)
		else:
			tf.truncated_normal(shape, mean=0, stddev=stddev)
	raise ValueError('Distribution must be either "uniform" or "normal".')

def constant_initializer(value, shape, lambda_initializer=True):
	if lambda_initializer:
		return np.full(shape, value).astype(np.float32)
	else:
		return tf.constant(value, tf.float32, shape)

def get_num_channels(x):
	"""
	:param x: an input tensor with shape [batch_size, ..., num_channels]
	:return: the number of channels of x
	"""
	return int(x.get_shape()[-1])

def get_spatial_rank(x):
	"""
	:param x: an input tensor with shape [batch_size, ..., num_channels]
	:return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
	"""
	return len(x.get_shape()) - 2

def get_spatial_size(x):
	"""
	:param x: an input tensor with shape [batch_size, ..., num_channels]
	:return: The spatial shape of x, excluding batch_size and num_channels.
	"""
	return x.get_shape()[1:-1]

def convolution(x, filter, padding='SAME', strides=None, dilation_rate=None, initializer="XAVIER"):
	w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
	b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-1]))

	return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b

def deconvolution(x, filter, output_shape, strides, padding='SAME'):
	w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
	b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-2]))

	spatial_rank = get_spatial_rank(x)
	if spatial_rank == 2:
		return tf.nn.conv2d_transpose(x, w, output_shape, strides, padding) + b
	if spatial_rank == 3:
		return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
	raise ValueError('Only 2D and 3D images supported.')

# More complex blocks
# down convolution
def down_convolution(x, factor, kernel_size):
	num_channels = get_num_channels(x)
	spatial_rank = get_spatial_rank(x)
	strides = spatial_rank * [factor]
	filter = kernel_size + [num_channels, num_channels * factor]
	x = convolution(x, filter, strides=strides)
	return x


# up convolution
def up_convolution(x, output_shape, factor, kernel_size):
	num_channels = get_num_channels(x)
	spatial_rank = get_spatial_rank(x)
	strides = [1] + spatial_rank * [factor] + [1]
	filter = kernel_size + [num_channels // factor, num_channels]
	x = deconvolution(x, filter, output_shape, strides=strides)
	return x

# parametric leaky relu
def prelu(x):
	alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
	return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

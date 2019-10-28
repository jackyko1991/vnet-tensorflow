import tensorflow as tf
from layers2 import *

class UNet(object):
	def __init__(self,
		num_output_channels,
		dropout_rate=0.01,
		num_channels=4,
		num_levels=4,
		num_convolutions=2,
		bottom_convolutions=2,
		is_training=True,
		activation_fn="relu"
		):
		"""
			Implements UNet architecture https://arxiv.org/abs/1505.04597
			:param num_output_channel: Number of output image channel
			:param dropout_rate: Dropout rate, set to 0.0 if not training or if no dropout is desired.
			:param num_channels: The number of output channels in the first level, this will be doubled every level.
			:param num_levels: The number of levels in the network. Default is 4 as in the paper.
			:param num_convolutions: An array with the number of convolutions at each level.
			:param bottom_convolutions: The number of convolutions at the bottom level of the network.
			:param activation_fn: The activation function.
		"""
		self.num_output_channels = num_output_channels
		self.dropout_rate = dropout_rate
		self.num_channels = num_channels
		self.num_levels = num_levels
		self.num_convolutions = num_convolutions
		self.bottom_convolutions = bottom_convolutions
		self.is_training = is_training
		self.train_phase = tf.placeholder(tf.bool,name="train_phase_placeholder")

		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		elif(activation_fn == "prelu"):
			self.activation_fn = prelu
		elif activation_fn == "lrelu":
			self.activation_fn = tf.nn.leaky_relu

	def convolution_block(self,layer_input, output_channel, num_convolutions, dropout_rate, activation_fn, is_training):
		x = layer_input
		input_channels = get_num_channels(x)
		spatial_rank = get_spatial_rank(x)

		for i in range(num_convolutions):
			with tf.variable_scope('conv_' + str(i+1)):
				if i == 0:
					if spatial_rank == 2:
						x = convolution(x, [3,3, input_channels, output_channel])
					else:
						x = convolution(x, [3,3,3, input_channels, output_channel])
				else:
					if spatial_rank == 2:
						x = convolution(x, [3,3, output_channel, output_channel])
					else:
							x = convolution(x, [3,3,3, output_channel, output_channel])
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, training=is_training)
				x = activation_fn(x)
				x = tf.nn.dropout(x, rate=dropout_rate)
		return x

	def convolution_block_2(self,layer_input, fine_grained_features, num_convolutions, dropout_rate, activation_fn, is_training):
		x = tf.concat((layer_input, fine_grained_features), axis=-1)
		x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)

		num_channels = get_num_channels(layer_input)

		spatial_rank = get_spatial_rank(x)

		#x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		for i in range(num_convolutions):
			with tf.variable_scope('conv_'+ str(i+1)):
				if i ==0:
					if spatial_rank == 2:
						x = convolution(x, [3,3, num_channels*2, num_channels])
					else:
						x = convolution(x, [3,3,3, num_channels*2, num_channels])
				else:
					if spatial_rank == 2:
						x = convolution(x, [3,3, num_channels, num_channels])
					else:
						x = convolution(x, [3,3,3, num_channels, num_channels])
			x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
			x = activation_fn(x)
			x = tf.nn.dropout(x, rate=dropout_rate)

		# x = layer_input + fine_grained_features
		# num_channels = get_num_channels(layer_input)

		# #x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		# for i in range(num_convolutions):
		# 	with tf.variable_scope('conv_'+ str(i+1)):
		# 		x = convolution(x, [3,3,3, num_channels, num_channels])
		# 	x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		# 	x = activation_fn(x)
		# 	x = tf.nn.dropout(x, rate=dropout_rate)

		return x

	def GetNetwork(self, x):
		dropout_rate = self.dropout_rate if self.is_training else 0.0
		# if the input has more than 1 channel then it has to be expanded because broadcasting only works for 1 channel input
		# channel
		input_channels = get_num_channels(x)

		# downward encoding
		features = []
		# x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, training=self.train_phase)

		for l in range(self.num_levels):
			with tf.variable_scope('unet/encoder/level_' + str(l+1)):
				x = self.convolution_block(x, self.num_channels*(2**(l)),self.num_convolutions, dropout_rate, activation_fn=self.activation_fn, is_training=self.train_phase)
				features.append(x)
				with tf.variable_scope('max_pooling'):
					spatial_rank = get_spatial_rank(x)
					if spatial_rank == 2:
						x = tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
					elif spatial_rank == 3:
						x = tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
					else:
						raise ValueError('Only 2D and 3D images supported.')

		# bottom level
		with tf.variable_scope('unet/bottom_level'):
			x = self.convolution_block(x, self.num_channels*(2**(l+1)), self.bottom_convolutions, dropout_rate, activation_fn=self.activation_fn, is_training=self.train_phase)

		# upward decoding
		for l in reversed(range(self.num_levels)):
			with tf.variable_scope('unet/decoder/level_' + str(l+1)):
				f = features[l]
				with tf.variable_scope('up_convolution'):
					if spatial_rank == 2:
						x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2,2])
					else:
						x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2,2,2])
					x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
					x = self.activation_fn(x)
					x = tf.nn.dropout(x, rate=dropout_rate)
				x = self.convolution_block_2(x, f, self.num_convolutions, dropout_rate, activation_fn=self.activation_fn, is_training=self.train_phase)

		# final output
		with tf.variable_scope('unet/output'):
			if spatial_rank == 2:
				logits = convolution(x, [1,1, self.num_channels,self.num_output_channels])
			else:
					logits = convolution(x, [1,1,1, self.num_channels,self.num_output_channels])
			logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)

		return logits

class Dense(object):
	def __init__(self,
		num_output_channels,
		dropout_rate=0.01,
		num_levels=2,
		is_training=True,
		activation_fn="relu"
		):
		"""
			Implements UNet architecture https://arxiv.org/abs/1505.04597
			:param num_output_channel: Number of output image channel
			:param dropout_rate: Dropout rate, set to 0.0 if not training or if no dropout is desired.
			:param num_channels: The number of output channels in the first level, this will be doubled every level.
			:param num_levels: The number of levels in the network. Default is 4 as in the paper.
			:param num_convolutions: An array with the number of convolutions at each level.
			:param bottom_convolutions: The number of convolutions at the bottom level of the network.
			:param activation_fn: The activation function.
		"""
		self.num_output_channels = num_output_channels
		self.dropout_rate = dropout_rate
		self.num_levels = num_levels
		self.is_training = is_training
		self.train_phase = tf.placeholder(tf.bool,name="train_phase_placeholder")

		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		elif(activation_fn == "prelu"):
			self.activation_fn = prelu
		elif activation_fn == "lrelu":
			self.activation_fn = tf.nn.leaky_relu

	def GetNetwork(self,x):
		input_tensor = x

		spatial_rank = get_spatial_rank(x)
		if spatial_rank == 2:
			x = tf.reshape(x, [-1, input_tensor.get_shape()[1]*input_tensor.get_shape()[2]*input_tensor.get_shape()[3]])
		elif spatial_rank == 3:
			x = tf.reshape(x, [-1, input_tensor.get_shape()[1]*input_tensor.get_shape()[2]*input_tensor.get_shape()[3]*input_tensor.get_shape()[4]])
		else:
			raise ValueError('Only 2D and 3D images supported.')
		x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)

		for l in range(self.num_levels):
			x=tf.layers.dense(x,units=128,activation=self.activation_fn)
			x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
		
		if spatial_rank == 2:
			x=tf.layers.dense(inputs=x,units=input_tensor.get_shape()[1]*input_tensor.get_shape()[2]*self.num_output_channels,activation=None)
			logits= tf.reshape(x, [-1, input_tensor.get_shape()[1],input_tensor[2],self.num_output_channels])
		elif spatial_rank == 3:
			x=tf.layers.dense(inputs=x,units=input_tensor.get_shape()[1]*input_tensor.get_shape()[2]*input_tensor.get_shape()[3]*self.num_output_channels,activation=None)
			x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
			logits= tf.reshape(x, [-1, input_tensor.get_shape()[1], input_tensor.get_shape()[2], input_tensor.get_shape()[3],self.num_output_channels])

		return logits

class VNet(object):
	def __init__(self,
				 num_classes,
				 dropout_rate=0.01,
				 num_channels=16,
				 num_levels=4,
				 num_convolutions=(1, 2, 3, 3),
				 bottom_convolutions=3,
				 is_training = True,
				 activation_fn="relu"):
		"""
		Implements VNet architecture https://arxiv.org/abs/1606.04797
		:param num_classes: Number of output classes.
		:param dropout_rate: Dropout_rate: Dropout rate, set to 0.0 if not training or if no dropout is desired.
		:param num_channels: The number of output channels in the first level, this will be doubled every level.
		:param num_levels: The number of levels in the network. Default is 4 as in the paper.
		:param num_convolutions: An array with the number of convolutions at each level.
		:param bottom_convolutions: The number of convolutions at the bottom level of the network.
		:param activation_fn: The activation function.
		"""
		self.num_classes = num_classes
		self.dropout_rate = dropout_rate
		self.num_channels = num_channels
		assert num_levels == len(num_convolutions)
		self.num_levels = num_levels
		self.num_convolutions = num_convolutions
		self.bottom_convolutions = bottom_convolutions
		self.is_training = is_training
		self.train_phase = tf.placeholder(tf.bool,name="train_phase_placeholder")

		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		elif(activation_fn == "prelu"):
			self.activation_fn = prelu
		elif activation_fn == "lrelu":
			self.activation_fn = tf.nn.leaky_relu

	def GetNetwork(self, x):
		spatial_rank = get_spatial_rank(x)

		dropout_rate = self.dropout_rate if self.is_training else 0.0
		# if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
		# channel
		input_channels = int(x.get_shape()[-1])
		with tf.variable_scope('vnet/input_layer'):
			if input_channels == 1:
				if spatial_rank == 2:
					x = tf.tile(x, [1, 1, 1, self.num_channels])
				else:
					x = tf.tile(x, [1, 1, 1, 1, self.num_channels])
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
			else:
				if spatial_rank == 2:
					x = convolution(x, [5, 5, input_channels, self.num_channels])
				else:
					x = convolution(x, [5, 5, 5, input_channels, self.num_channels])
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
				x = self.activation_fn(x)

		features = list()

		for l in range(self.num_levels):
			with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
				x = self.convolution_block(x, self.num_convolutions[l], dropout_rate, activation_fn=self.activation_fn, is_training=self.train_phase)
				features.append(x)
				with tf.variable_scope('down_convolution'):
					if spatial_rank == 2:
						x = down_convolution(x, factor=2, kernel_size=[2, 2])
					else:
						x = down_convolution(x, factor=2, kernel_size=[2, 2, 2])
					x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
					x = self.activation_fn(x)

		with tf.variable_scope('vnet/bottom_level'):
			x = self.convolution_block(x, self.bottom_convolutions, dropout_rate, activation_fn=self.activation_fn, is_training=self.train_phase)

		for l in reversed(range(self.num_levels)):
			with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
				f = features[l]
				with tf.variable_scope('up_convolution'):
					if spatial_rank == 2:
						x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2])
					else:
						x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2, 2])
					x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)
					x = self.activation_fn(x)

				x = self.convolution_block_2(x, f, self.num_convolutions[l], dropout_rate, activation_fn=self.activation_fn, is_training=self.train_phase)

		with tf.variable_scope('vnet/output_layer'):
			if spatial_rank == 2:
				logits = convolution(x, [1, 1, self.num_channels, self.num_classes])
			else:
				logits = convolution(x, [1, 1, 1, self.num_channels, self.num_classes])
			logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.train_phase)

		return logits

	def convolution_block(self,layer_input, num_convolutions, dropout_rate, activation_fn, is_training):
		x = layer_input
		n_channels = get_num_channels(x)
		spatial_rank = get_spatial_rank(layer_input)
		for i in range(num_convolutions):
			with tf.variable_scope('conv_' + str(i+1)):
				if spatial_rank == 2:
					x = convolution(x, [5, 5, n_channels, n_channels])
				else:
					x = convolution(x, [5, 5, 5, n_channels, n_channels])
				if i == num_convolutions - 1:
					x = x + layer_input
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				x = activation_fn(x)
				x = tf.nn.dropout(x, rate=dropout_rate)
		return x

	def convolution_block_2(self,layer_input, fine_grained_features, num_convolutions, dropout_rate, activation_fn, is_training):
		x = tf.concat((layer_input, fine_grained_features), axis=-1)
		n_channels = get_num_channels(layer_input)
		spatial_rank = get_spatial_rank(layer_input)
		if num_convolutions == 1:
			with tf.variable_scope('conv_' + str(1)):
				if spatial_rank == 2:
					x = convolution(x, [5, 5, n_channels * 2, n_channels])
				else:
					x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				layer_input = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				x = x + layer_input
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				x = activation_fn(x)
				x = tf.nn.dropout(x, rate=dropout_rate)
			return x

		with tf.variable_scope('conv_' + str(1)):
			if spatial_rank == 2:
				x = convolution(x, [5, 5, n_channels * 2, n_channels])
			else:
				x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
			x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
			x = activation_fn(x)
			x = tf.nn.dropout(x, rate=dropout_rate)

		for i in range(1, num_convolutions):
			with tf.variable_scope('conv_' + str(i+1)):
				if spatial_rank == 2:
					x = convolution(x, [5, 5, n_channels, n_channels])
				else:
					x = convolution(x, [5, 5, 5, n_channels, n_channels])
				# x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				layer_input = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				if i == num_convolutions - 1:
					x = x + layer_input
				x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
				x = activation_fn(x)
				x = tf.nn.dropout(x, rate=dropout_rate)

		return x
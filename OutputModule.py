# MIT License
#
# Copyright (c) 2019 Jacky Ka Long, Ko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf

def init_weight(shape):
	w = tf.truncated_normal(shape= shape, mean=0, stddev =0.1)
	return tf.Variable(w)

def init_bias(shape):
	b = tf.zeros(shape)
	return tf.Variable(b)

class OutputModule(object):
	"""
	Output module model with residual block design
	"""

	def __init__(self,
		num_classes,
		num_channels=64,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0):
		"""
		Implements Resnet3D in 3D
		:param num_classes: Number of output classes.
		:param num_classes: Number of feature channels.
		:param is_training: Set network in training mode
		:param activation_fn: The activation function.
		:param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
		"""
		self.num_classes = num_classes
		self.num_channels = num_channels
		self.is_training = is_training
		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		else:
			print("Invalid activation function")
			exit()
		self.keep_prob = keep_prob

	def Conv3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		return conv

	def ConvActivate3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		return conv

	def residual_block(self, input_tensor, channels, output_activation=True,is_training=True):
		paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
		input_tensor_padded = tf.pad(input_tensor, paddings, "CONSTANT")
		input_channels = int(input_tensor.get_shape()[-1])
		conv1Filter_shape = [3,3,3,input_channels,channels]
		conv1 = self.ConvActivate3d_block(input_tensor_padded, conv1Filter_shape, is_training = self.is_training)
		conv1 = tf.pad(conv1, paddings, "CONSTANT")
		conv2Filter_shape = [3,3,3,channels,channels]
		conv2 = self.Conv3d_block(conv1, conv2Filter_shape, is_training = self.is_training)

		# residual branch
		conv_up_W = init_weight([1,1,1,input_channels,channels])
		conv_up_B = init_bias(channels)
		input_tensor_up_conv = tf.nn.conv3d(input_tensor, conv_up_W, strides = [1,1,1,1,1], padding ='VALID') + conv_up_B

		output = tf.add(conv2, input_tensor_up_conv)
		output = tf.layers.batch_normalization(output, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		if output_activation:
			output = self.activation_fn(output)
		output = tf.nn.dropout(output, self.keep_prob)
		return output

	def GetNetwork(self, input):
		with tf.variable_scope('output/encoder'):
			layer1_resblock1 = self.residual_block(input, self.num_channels, True, self.is_training)
			layer1_resblock2 = self.residual_block(layer1_resblock1, self.num_channels, True, self.is_training)
			layer1_resblock3 = self.residual_block(layer1_resblock2, self.num_channels, True, self.is_training)

		with tf.variable_scope('output/output'):
			logits = self.Conv3d_block(layer1_resblock3, [1, 1, 1, self.num_channels, self.num_classes])

		return logits


if __name__=="__main__":
	main()
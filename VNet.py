# MIT License
#
# Copyright (c) 2018 Miguel Monteiro
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
from Layers import convolution, down_convolution, up_convolution, get_num_channels,prelu


def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn, is_training):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn, is_training):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    n_channels = get_num_channels(layer_input)
    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            layer_input = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
        return x

    with tf.variable_scope('conv_' + str(1)):
        x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
        x = activation_fn(x)
        x = tf.nn.dropout(x, keep_prob)

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            layer_input = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)

    return x


class VNet(object):
    def __init__(self,
                 num_classes,
                 keep_prob=1.0,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 is_training = True,
                 activation_fn="relu"):
        """
        Implements VNet architecture https://arxiv.org/abs/1606.04797
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.is_training = is_training

        if (activation_fn == "relu"):
            self.activation_fn = tf.nn.relu
        elif(activation_fn == "prelu"):
            self.activation_fn = prelu

    def network_fn(self, x):

        keep_prob = self.keep_prob if self.is_training else 1.0
        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
        # channel
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope('vnet/input_layer'):
            if input_channels == 1:
                x = tf.tile(x, [1, 1, 1, 1, self.num_channels])
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

            else:
                x = convolution(x, [5, 5, 5, input_channels, self.num_channels])
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)
                x = self.activation_fn(x)

        features = list()

        for l in range(self.num_levels):
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                x = convolution_block(x, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn, is_training=self.is_training)
                features.append(x)
                with tf.variable_scope('down_convolution'):
                    x = down_convolution(x, factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)
                    x = self.activation_fn(x)

        with tf.variable_scope('vnet/bottom_level'):
            x = convolution_block(x, self.bottom_convolutions, keep_prob, activation_fn=self.activation_fn, is_training=self.is_training)

        for l in reversed(range(self.num_levels)):
            with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
                f = features[l]
                with tf.variable_scope('up_convolution'):
                    x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)
                    x = self.activation_fn(x)

                x = convolution_block_2(x, f, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn, is_training=self.is_training)

        with tf.variable_scope('vnet/output_layer'):
            logits = convolution(x, [1, 1, 1, self.num_channels, self.num_classes])
            logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

        return logits
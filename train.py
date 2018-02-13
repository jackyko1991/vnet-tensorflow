from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import NiftiDataInput as NDI

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_integer('batch_size',16,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',128,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',2000
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('train_dir', './tmp/train_log',
    """Directory where to write training event logs """)

def placeholder_inputs(input_batch_shape, output_batch_shape):
    """Generate placeholder variables to represent the the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded ckpt in the .run() loop, below.
    Args:
        patch_shape: The patch_shape will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test ckpt sets.
    # batch_size = -1

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape)
    labels_placeholder = tf.placeholder(tf.float32, shape=output_batch_shape)   
   
    return images_placeholder, labels_placeholder

def train():
    """Train the Vnet model"""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # patch_shape(batch_size, height, width, depth, channels)
        input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
        output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
        
        image_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape)

        # Get images and labels
        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            train_data_dir = os.path.join(FLAGS.data_dir,'training')
            test_data_dir = os.path.join(FLAGS.data_dir,'testing')
            images_train, labels_train = NDI.augumented_inputs(train_data_dir,input_batch_shape,output_batch_shape)

def main(argv=None):
    # clear training log directory
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
train()

if __name__=='__main__':
    tf.app.run()
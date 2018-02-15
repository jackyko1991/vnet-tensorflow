from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import NiftiDataset
import os
import VNet
import math
import datetime

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_integer('batch_size',5,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',128,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',2000,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('train_dir', './tmp/train_log',
    """Directory where to write training event logs """)
tf.app.flags.DEFINE_string('tensorboard_dir', './tmp/tensorboard',
    """Directory where to write tensorboard summary """)
tf.app.flags.DEFINE_float('init_learning_rate',0.1,
    """Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',0.01,
    """Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',100,
    """Number of epoch before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',10,
    """Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1,
    """Checkpoint save interval (epochs)""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
    """Directory where to write checkpoint""")
tf.app.flags.DEFINE_float('drop_ratio',0.5,
    """Probability to drop a cropped area if the label is empty. All empty patches will be droped for 0 and accept all cropped patches if set to 1""")
tf.app.flags.DEFINE_integer('min_pixel',10,
    """Minimum non-zero pixels in the cropped label""")

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
    labels_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape)   
   
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
        train_data_dir = os.path.join(FLAGS.data_dir,'training')
        test_data_dir = os.path.join(FLAGS.data_dir,'testing')
        # support multiple image input, but here only use single channel, label file should be a single file with different classes
        image_filename = 'img.nii.gz'
        label_filename = 'label.nii.gz'

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            # create transformations to image and labels
            transforms = [
                NiftiDataset.Normalization(),
                NiftiDataset.Resample(0.4356),
                NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
                NiftiDataset.RandomNoise()
                ]

            TrainDataset = NiftiDataset.NiftiDataset(
                data_dir=train_data_dir,
                image_filename=image_filename,
                label_filename=label_filename,
                transforms=transforms,
                train=True
                )

            trainDataset = TrainDataset.get_dataset()
            trainDataset = trainDataset.shuffle(buffer_size=10)
            print("batch size =",FLAGS.batch_size)
            trainDataset = trainDataset.batch(FLAGS.batch_size)

        iterator = trainDataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        # # Initialize the model
        # logits = VNet.v_net(image_placeholder,input_batch_shape[4],output_batch_shape[4])

        # # Exponential decay learning rate
        # train_batches_per_epoch = math.ceil(train_generator.data_size/FLAGS.batch_size)
        # decay_steps = train_batches_per_epoch*FLAGS.decay_steps

        # learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate,
        #     global_step,
        #     decay_steps,
        #     FLAGS.decay_factor,
        #     staircase=True)
        # tf.summary.scalar('learning_rate', learning_rate)

        # # Op for calculating loss
        # with tf.name_scope("cross_entropy"):
        #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_placeholder))
        # tf.summary.scalar('loss',loss)

        # Training Op
        # with tf.name_scope("training"):
        #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #     train_op = optimizer.minimize(
        #         loss=loss,
        #         global_step=global_step)

        # training cycle
        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            print("{} Start training...".format(datetime.datetime.now()))

            # loop over epochs
            for epoch in range(FLAGS.epochs):
                print("{} Epoch {} starts".format(datetime.datetime.now(),epoch+1))

                # # initialize iterator
                # sess.run(iterator.initializer)
                while True:
                    try:
                        [image, label] = sess.run(next_element)
                        print(image.shape)
                        print(label.shape)
                        # print(tf.shape(sess.run(next_element)))
                    except tf.errors.OutOfRangeError:
                        break

                    #     
        #     for epoch in range(FLAGS.epochs):
        #         
            
        #         step=0
        #         while step < train_batches_per_epoch:
        #             # Get a batch of image and labels
        #             image_batch, label_batch = train_generator.next_batch()
        




        
        # # Evaluation op: Accuracy of model
        # with tf.name_scope("accuracy"):
        #     correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels_placeholder,1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # # add accuracy to summary
        # tf.summary.scalar('accuracy', accuracy)

        # # Merge all summaries
        # merged_summary = tf.summary.merge_all()

        # # Initialize summary filewriter
        # if not os.path.exists(FLAGS.tensorboard_dir):
        #     os.mkdir(FLAGS.tensorboard_dir)
        # writer = tf.summary.FileWriter(FLAGS.tensorboard_dir)

        # # Initialize svaer for storing model checkpoints
        # saver = tf.train.Saver()

        # # Start tensorflow session
        # with tf.Session() as sess:
        #     # Initialize all variables
        #     sess.run(tf.global_variables_initializer())

        #     # add model graph to tensorboard
        #     writer.add_graph(sess.graph)

        #     print("{} Start training...".format(datetime.datetime.now()))
        #     print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(),FLAGS.tensorboard_dir))

        #     # loop over epochs
        #     for epoch in range(FLAGS.epochs):
        #         print("{} Epoch {} starts".format(datetime.datetime.now(),epoch+1))
            
        #         step=0
        #         while step < train_batches_per_epoch:
        #             # Get a batch of image and labels
        #             image_batch, label_batch = train_generator.next_batch()



def main(argv=None):
    # clear training log directory
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # clear checkpoint directory
    if tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()

if __name__=='__main__':
    tf.app.run()
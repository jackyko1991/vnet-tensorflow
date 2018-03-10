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
from tensorflow.python.tools import inspect_checkpoint as chkp

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_integer('batch_size',1,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',128,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',2000,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',0.0001,
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
tf.app.flags.DEFINE_bool('restore_training',True,
    """Restore training from last checkpoint""")
tf.app.flags.DEFINE_float('drop_ratio',0.5,
    """Probability to drop a cropped area if the label is empty. All empty patches will be droped for 0 and accept all cropped patches if set to 1""")
tf.app.flags.DEFINE_integer('min_pixel',10,
    """Minimum non-zero pixels in the cropped label""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',5,
    """Number of elements used in shuffle buffer""")

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

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    labels_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")   
   
    return images_placeholder, labels_placeholder

def train():
    """Train the Vnet model"""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # patch_shape(batch_size, height, width, depth, channels)
        input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
        output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
        
        images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape)

        images_log = tf.cast(images_placeholder[:,:,:,int(FLAGS.patch_layer/2),:], dtype=tf.uint8)
        labels_log = tf.cast(tf.scalar_mul(255,labels_placeholder[:,:,:,int(FLAGS.patch_layer/2),:]), dtype=tf.uint8)

        tf.summary.image("image", images_log,max_outputs=FLAGS.batch_size)
        tf.summary.image("label", labels_log,max_outputs=FLAGS.batch_size)

        # Get images and labels
        train_data_dir = os.path.join(FLAGS.data_dir,'training')
        test_data_dir = os.path.join(FLAGS.data_dir,'testing')
        # support multiple image input, but here only use single channel, label file should be a single file with different classes
        image_filename = 'img.nii.gz'
        label_filename = 'label.nii.gz'

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            # create transformations to image and labels
            trainTransforms = [
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
                transforms=trainTransforms,
                train=True
                )
            
            trainDataset = TrainDataset.get_dataset()
            trainDataset = trainDataset.shuffle(buffer_size=5)
            trainDataset = trainDataset.batch(FLAGS.batch_size)

            testTransforms = [
                NiftiDataset.Normalization(),
                NiftiDataset.Resample(0.4356),
                NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel)
                ]

            TestDataset = NiftiDataset.NiftiDataset(
                data_dir=train_data_dir,
                image_filename=image_filename,
                label_filename=label_filename,
                transforms=testTransforms,
                train=True
            )

            testDataset = TestDataset.get_dataset()
            testDataset = testDataset.shuffle(buffer_size=5)
            testDataset = testDataset.batch(FLAGS.batch_size)
            
        train_iterator = trainDataset.make_initializable_iterator()
        next_element_train = train_iterator.get_next()

        test_iterator = testDataset.make_initializable_iterator()
        next_element_test = test_iterator.get_next()

        # Initialize the model
        with tf.name_scope("vnet"):
            logits = VNet.v_net(images_placeholder,input_channels = input_batch_shape[4], output_channels =2)

        logits_log_0 = tf.cast(logits[:,:,:,int(FLAGS.patch_layer/2):int(FLAGS.patch_layer/2)+1,0], dtype=tf.uint8)
        logits_log_1 = tf.cast(logits[:,:,:,int(FLAGS.patch_layer/2):int(FLAGS.patch_layer/2)+1,1], dtype=tf.uint8)
        tf.summary.image("logits_0", logits_log_0,max_outputs=FLAGS.batch_size)
        tf.summary.image("logits_1", logits_log_1,max_outputs=FLAGS.batch_size)

        # # Exponential decay learning rate
        # train_batches_per_epoch = math.ceil(TrainDataset.data_size/FLAGS.batch_size)
        # decay_steps = train_batches_per_epoch*FLAGS.decay_steps

        with tf.name_scope("learning_rate"):
            learning_rate = FLAGS.init_learning_rate
        #     learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate,
        #         global_step,
        #         decay_steps,
        #         FLAGS.decay_factor,
        #         staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # softmax op for probability layer
        with tf.name_scope("softmax"):
            softmax_op = tf.nn.softmax(logits,name="softmax")
        softmax_log_0 = tf.cast(tf.scalar_mul(255,softmax_op[:,:,:,int(FLAGS.patch_layer/2):int(FLAGS.patch_layer/2)+1,0]), dtype=tf.uint8)
        softmax_log_1 = tf.cast(tf.scalar_mul(255,softmax_op[:,:,:,int(FLAGS.patch_layer/2):int(FLAGS.patch_layer/2)+1,1]), dtype=tf.uint8)

        tf.summary.image("softmax_0", softmax_log_0,max_outputs=FLAGS.batch_size)
        tf.summary.image("softmax_1", softmax_log_1,max_outputs=FLAGS.batch_size)

        # Op for calculating loss
        with tf.name_scope("cross_entropy"):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder, 
                squeeze_dims=[4])))
        tf.summary.scalar('loss',loss_op)

        # Argmax Op to generate label from logits
        with tf.name_scope("predicted_label"):
            pred = tf.argmax(logits, axis=4 , name="prediction")
        pred_log = tf.cast(tf.scalar_mul(255,pred[:,:,:,int(FLAGS.patch_layer/2):int(FLAGS.patch_layer/2)+1]), dtype=tf.uint8)
        tf.summary.image("pred", pred_log,max_outputs=FLAGS.batch_size)

        # Training Op
        with tf.name_scope("training"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.init_learning_rate)
            train_op = optimizer.minimize(
                loss=loss_op,
                global_step=global_step)

        # saver
        print("Setting up Saver...")
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir ,"checkpoint")

        # epoch checkpoint manipulation
        start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
        # start_epoch_inc = start_epoch.assign(start_epoch+1)

        # training cycle
        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            print("{}: Start training...".format(datetime.datetime.now()))

            # summary writer for tensorboard
            train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

            # restore from checkpoint
            if FLAGS.restore_training:
                # check if checkpoint exists
                if os.path.exists(checkpoint_prefix+"-latest"):
                    print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
                    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename="checkpoint-latest")
                    saver.restore(sess, latest_checkpoint_path)
            
            # print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))

            # loop over epochs
            for epoch in np.arange(start_epoch.eval(), FLAGS.epochs):
                # initialize iterator in each new epoch
                sess.run(train_iterator.initializer)
                sess.run(test_iterator.initializer)
                print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

                # training phase
                while True:
                    try:
                        [image, label] = sess.run(next_element_train)

                        image = image[:,:,:,:,np.newaxis]
                        label = label[:,:,:,:,np.newaxis]
                        
                        train, summary = sess.run([train_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError:
                        # start_epoch_inc.op.run()
                        # print(start_epoch.eval())
                        # save the model at end of each epoch training
                        print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,FLAGS.checkpoint_dir))
                        saver.save(sess, checkpoint_prefix, 
                            global_step=tf.train.global_step(sess, global_step),
                            latest_filename="checkpoint-latest")
                        print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
                        break
                
                # testing phase
                print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
                while True:
                    try:
                        [image, label] = sess.run(next_element_test)

                        image = image[:,:,:,:,np.newaxis]
                        label = label[:,:,:,:,np.newaxis]
                        
                        loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

                    except tf.errors.OutOfRangeError:
                        break

        # close tensorboard summary writer
        train_summary_writer.close()
        test_summary_writer.close()


        
        # # Evaluation op: Accuracy of model
        # with tf.name_scope("accuracy"):
        #     correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels_placeholder,1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        # # add accuracy to summary
        # tf.summary.scalar('accuracy', accuracy)

def main(argv=None):
    if not FLAGS.restore_training:
        # clear log directory
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

        # clear checkpoint directory
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    train()

if __name__=='__main__':
    tf.app.run()
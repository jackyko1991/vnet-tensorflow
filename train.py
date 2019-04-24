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
import attention
import OutputModule
import json

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data_SWAN',
	"""Directory of stored data.""")
tf.app.flags.DEFINE_string('config_json','./config.json',
	"""JSON file for filename configuration""")
tf.app.flags.DEFINE_integer('batch_size',1,
	"""Size of batch""")           
tf.app.flags.DEFINE_integer('patch_size',64,
	"""Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',64,
	"""Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',999999999,
	"""Number of epochs for training""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
	"""Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-2,
	"""Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',1.0,
	"""Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',100,
	"""Number of epoch before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',10,
	"""Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1,
	"""Checkpoint save interval (epochs)""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
	"""Directory where to write checkpoint""")
tf.app.flags.DEFINE_string('model_dir','./tmp/model',
	"""Directory to save model""")
tf.app.flags.DEFINE_bool('restore_training',True,
	"""Restore training from last checkpoint""")
tf.app.flags.DEFINE_float('drop_ratio',0.001,
	"""Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1""")
tf.app.flags.DEFINE_integer('min_pixel',30,
	"""Minimum non-zero pixels in the cropped label""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',5,
	"""Number of elements used in shuffle buffer""")
tf.app.flags.DEFINE_string('loss_function','sorensen',
	"""Loss function used in optimization (xent, weight_xent, sorensen, jaccard)""")
tf.app.flags.DEFINE_string('attention_loss_function','l2',
	"""Loss function used in optimization (l2,abs)""")
tf.app.flags.DEFINE_string('optimizer','sgd',
	"""Optimization method (sgd, adam, momentum, nesterov_momentum)""")
tf.app.flags.DEFINE_float('momentum',0.5,
	"""Momentum used in optimization""")
tf.app.flags.DEFINE_bool('testing',True,
	"""Perform testing after each epoch""")
tf.app.flags.DEFINE_bool('attention',True,
	"""Perform testing after each epoch""")
tf.app.flags.DEFINE_bool('image_log',False,
	"""Perform testing after each epoch""")

# tf.app.flags.DEFINE_float('class_weight',0.15,
#     """The weight used for imbalanced classes data. Currently only apply on binary segmentation class (weight for 0th class, (1-weight) for 1st class)""")

def placeholder_inputs(input_batch_shape, output_batch_shape, attention=False):
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

	if attention:
		distmap_placeholder = tf.placeholder(tf.float32, shape=output_batch_shape, name="distmap_placeholder")   
		return images_placeholder, labels_placeholder, distmap_placeholder
	else:
		return images_placeholder, labels_placeholder

def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
	"""Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
	of two batch of data, usually be used for binary image segmentation
	i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

	Parameters
	-----------
	output : Tensor
		A distribution with shape: [batch_size, ....], (any dimensions).
	target : Tensor
		The target distribution, format the same with `output`.
	loss_type : str
		``jaccard`` or ``sorensen``, default is ``jaccard``.
	axis : tuple of int
		All dimensions are reduced, default ``[1,2,3]``.
	smooth : float
		This small value will be added to the numerator and denominator.
			- If both output and target are empty, it makes sure dice is 1.
			- If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

	Examples
	---------
	>>> outputs = tl.act.pixel_wise_softmax(network.outputs)
	>>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

	References
	-----------
	- `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

	"""

	inse = tf.reduce_sum(tf.multiply(output,target), axis=axis)

	if loss_type == 'jaccard':
		l = tf.reduce_sum(tf.multiply(output,output), axis=axis)
		r = tf.reduce_sum(tf.multiply(target,target), axis=axis)
	elif loss_type == 'sorensen':
		l = tf.reduce_sum(output, axis=axis)
		r = tf.reduce_sum(target, axis=axis)
	else:
		raise Exception("Unknown loss_type")
	## old axis=[0,1,2,3]
	# dice = 2 * (inse) / (l + r)
	# epsilon = 1e-5
	# dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
	## new haodong
	dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
	##
	dice = tf.reduce_mean(dice)
	return dice

def grayscale_to_rainbow(image):
	# grayscale to rainbow colormap, convert to HSV (H = reversed grayscale from 0:2/3, S and V are all 1)
	# then convert to RGB
	H = tf.squeeze((1. - image)*2./3., axis=-1)
	SV = tf.ones(H.get_shape())
	HSV = tf.stack([H,SV,SV], axis=3)
	RGB = tf.image.hsv_to_rgb(HSV)

	return RGB

def train():
	# read configuration file
	with open(FLAGS.config_json) as config_json:  
		config = json.load(config_json)

	"""Train the Vnet model"""
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		# patch_shape(batch_size, height, width, depth, channels)
		input_channel_num = len(config['TrainingSetting']['Data']['ImageFilenames'])

		input_batch_shape = (None, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, input_channel_num) 
		output_batch_shape = (None, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) 
		
		if FLAGS.attention:
			images_placeholder, labels_placeholder, distmap_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape, attention=True)
		else:
			images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape, attention=False)
		

		if FLAGS.image_log:
			for batch in range(FLAGS.batch_size):
				# plot images in tensorboard
				for image_channel in range(input_channel_num):
					images_log = tf.cast(images_placeholder[batch:batch+1,:,:,:,image_channel], dtype=tf.uint8)
					tf.summary.image(config['TrainingSetting']['Data']['ImageFilenames'][image_channel], tf.transpose(images_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

				if FLAGS.attention:
					# dist map will be plot in color
					distmap_log = grayscale_to_rainbow(tf.transpose(distmap_placeholder[batch:batch+1,:,:,:,0],[3,1,2,0]))
					distmap_log = tf.cast(tf.scalar_mul(255,distmap_log), dtype=tf.uint8)
					tf.summary.image("distmap", distmap_log,max_outputs=FLAGS.patch_layer)

				# plot labels
				labels_log = tf.cast(tf.scalar_mul(255,labels_placeholder[batch:batch+1,:,:,:,0]), dtype=tf.uint8)
				tf.summary.image("label", tf.transpose(labels_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
				
		# Get images and labels
		train_data_dir = os.path.join(FLAGS.data_dir,'training')
		test_data_dir = os.path.join(FLAGS.data_dir,'testing')
		# support multiple image input, but here only use single channel, label file should be a single file with different classes

		# Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
		with tf.device('/cpu:0'):
			# create transformations to image and labels
			trainTransforms = [
				NiftiDataset.StatisticalNormalization(2.5),
				# NiftiDataset.Normalization(),
				NiftiDataset.Resample((0.75,0.75,0.75)),
				NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
				# NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
				# NiftiDataset.ConfidenceCrop((FLAGS.patch_size*3, FLAGS.patch_size*3, FLAGS.patch_layer*3),(0.0001,0.0001,0.0001)),
				# NiftiDataset.BSplineDeformation(randomness=2),
				# NiftiDataset.ConfidenceCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),(0.5,0.5,0.5)),
				NiftiDataset.ConfidenceCrop2((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),rand_range=32,probability=0.5),
				NiftiDataset.RandomNoise()
				]

			TrainDataset = NiftiDataset.NiftiDataset(
				data_dir=train_data_dir,
				image_filenames=config['TrainingSetting']['Data']['ImageFilenames'],
				label_filename=config['TrainingSetting']['Data']['LabelFilename'],
				transforms=trainTransforms,
				train=True,
				distmap=FLAGS.attention
				)
			
			trainDataset = TrainDataset.get_dataset()
			trainDataset = trainDataset.shuffle(buffer_size=5)
			trainDataset = trainDataset.batch(FLAGS.batch_size,drop_remainder=True)

			if FLAGS.testing:
				# use random crop for testing
				testTransforms = [
					NiftiDataset.StatisticalNormalization(2.5),
					# NiftiDataset.Normalization(),
					NiftiDataset.Resample((0.75,0.75,0.75)),
					NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
					# NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel)
					# NiftiDataset.ConfidenceCrop((FLAGS.patch_size*2, FLAGS.patch_size*2, FLAGS.patch_layer*2),(0.0001,0.0001,0.0001)),
					# NiftiDataset.BSplineDeformation(),
					# NiftiDataset.ConfidenceCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),(0.75,0.75,0.75)),
					NiftiDataset.ConfidenceCrop2((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),rand_range=32,probability=0.5),
					]

				TestDataset = NiftiDataset.NiftiDataset(
					data_dir=test_data_dir,
					image_filenames=config['TrainingSetting']['Data']['ImageFilenames'],
					label_filename=config['TrainingSetting']['Data']['LabelFilename'],
					transforms=testTransforms,
					train=True,
					distmap=FLAGS.attention
				)

				testDataset = TestDataset.get_dataset()
				testDataset = testDataset.shuffle(buffer_size=5)
				testDataset = testDataset.batch(FLAGS.batch_size,drop_remainder=True)
			
		train_iterator = trainDataset.make_initializable_iterator()
		next_element_train = train_iterator.get_next()

		if FLAGS.testing:
			test_iterator = testDataset.make_initializable_iterator()
			next_element_test = test_iterator.get_next()

		# Initialize the model
		with tf.name_scope("vnet"):
			model = VNet.VNet(
				num_classes=2, # binary for 2
				keep_prob=1.0, # default 1
				num_channels=16, # default 16 
				num_levels=3,  # default 4
				num_convolutions=(1,2,2), # default (1,2,3,3), size should equal to num_levels
				bottom_convolutions=3, # default 3
				activation_fn="prelu") # default relu
			logits_vnet = model.network_fn(images_placeholder)

		if FLAGS.attention:
			with tf.name_scope("attention"):
				attentionModule = attention.AttentionModule(
					num_classes=2,
					is_training=True,
					activation_fn="relu",
					keep_prob=1.0)
				logits_attention = attentionModule.GetNetwork(logits_vnet)
				softmax_attention = tf.nn.softmax(logits_attention, name="softmax_attention")

			for batch in range(FLAGS.batch_size):
				softmax_att_log_0 = grayscale_to_rainbow(tf.transpose(softmax_attention[batch:batch+1,:,:,:,0],[3,1,2,0]))
				softmax_att_log_1 = grayscale_to_rainbow(tf.transpose(softmax_attention[batch:batch+1,:,:,:,1],[3,1,2,0]))
				softmax_att_log_0 = tf.cast(tf.scalar_mul(255,softmax_att_log_0), dtype=tf.uint8)
				softmax_att_log_1 = tf.cast(tf.scalar_mul(255,softmax_att_log_1), dtype=tf.uint8)
			   
				if FLAGS.image_log:
					tf.summary.image("softmax_attention_0", softmax_att_log_0,max_outputs=FLAGS.patch_layer)
					tf.summary.image("softmax_attention_1", softmax_att_log_1,max_outputs=FLAGS.patch_layer)

			with tf.name_scope("masked_vnet"):
				logits_masked = (1+softmax_attention)*logits_vnet

			with tf.name_scope("output"):
				outputModule = OutputModule.OutputModule(
					num_classes=2,
					is_training=True,
					activation_fn="relu",
					keep_prob=1.0)
				logits_output = outputModule.GetNetwork(logits_masked)
		else:
			logits_output = logits_vnet

		# for batch in range(FLAGS.batch_size):
		#     logits_max = tf.reduce_max(logits[batch:batch+1,:,:,:,:])
		#     logits_min = tf.reduce_min(logits[batch:batch+1,:,:,:,:])

		#     logits_log_0 = logits[batch:batch+1,:,:,:,0]
		#     logits_log_1 = logits[batch:batch+1,:,:,:,1]

		#     # normalize to 0-255 range
		#     logits_log_0 = tf.cast((logits_log_0-logits_min)*255./(logits_max-logits_min), dtype=tf.uint8)
		#     logits_log_1 = tf.cast((logits_log_1-logits_min)*255./(logits_max-logits_min), dtype=tf.uint8)

		#     tf.summary.image("logits_0", tf.transpose(logits_log_0,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
		#     tf.summary.image("logits_1", tf.transpose(logits_log_1,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

		# learning rate
		with tf.name_scope("learning_rate"):
			learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step,
				FLAGS.decay_steps,FLAGS.decay_factor,staircase=False)
		tf.summary.scalar('learning_rate', learning_rate)

		# softmax op for masked logit layer
		with tf.name_scope("softmax"):
			softmax_op = tf.nn.softmax(logits_output,name="softmax")

		if FLAGS.image_log:
			for batch in range(FLAGS.batch_size):
				softmax_log_0 = grayscale_to_rainbow(tf.transpose(softmax_op[batch:batch+1,:,:,:,0],[3,1,2,0]))
				softmax_log_1 = grayscale_to_rainbow(tf.transpose(softmax_op[batch:batch+1,:,:,:,1],[3,1,2,0]))
				softmax_log_0 = tf.cast(tf.scalar_mul(255,softmax_log_0), dtype=tf.uint8)
				softmax_log_1 = tf.cast(tf.scalar_mul(255,softmax_log_1), dtype=tf.uint8)
			   
				tf.summary.image("softmax_0", softmax_log_0,max_outputs=FLAGS.patch_layer)
				tf.summary.image("softmax_1", softmax_log_1,max_outputs=FLAGS.patch_layer)

		# Op for calculating loss
		with tf.name_scope("loss"):
			if (FLAGS.loss_function == "xent"):
				loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=logits_output,
					labels=tf.squeeze(labels_placeholder, 
					squeeze_dims=[4])))
			elif(FLAGS.loss_function == "weighted_cross_entropy"):
				class_weights = tf.constant([1.0, 1.0])

				# deduce weights for batch samples based on their true label
				onehot_labels = tf.one_hot(tf.squeeze(labels_placeholder,squeeze_dims=[4]),depth = 2)

				weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
				# compute your (unweighted) softmax cross entropy loss
				unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=logits_masked,
					labels=tf.squeeze(labels_placeholder, 
					squeeze_dims=[4]))
				# apply the weights, relying on broadcasting of the multiplication
				weighted_loss = unweighted_loss * weights
				# reduce the result to get your final loss
				loss_op = tf.reduce_mean(weighted_loss)
			elif(FLAGS.loss_function == "sorensen"):
				# Dice Similarity, currently only for binary segmentation, here we provide two calculation methods, first one is closer to classical dice formula
				sorensen = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='sorensen')
				# sorensen = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=2),dtype=tf.float32), loss_type='sorensen', axis=[1,2,3])
				loss_op = 1. - sorensen
			elif(FLAGS.loss_function == "jaccard"):
				# Dice Similarity, currently only for binary segmentation, here we provide two calculation methods, first one is closer to classical dice formula
				# jaccard = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='jaccard')
				jaccard = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=2),dtype=tf.float32), loss_type='jaccard', axis=[1,2,3])
				loss_op = 1. - jaccard
		tf.summary.scalar('loss',loss_op)

		if FLAGS.attention:
			# attention loss function
			with tf.name_scope("attention_loss"):
				if (FLAGS.attention_loss_function == "l2"):
					distmap_0 = 1. - tf.squeeze(distmap_placeholder,axis=-1)
					distmap_1 = tf.squeeze(distmap_placeholder,axis=-1)
					# distmap = tf.stack([distmap_0,distmap_1],axis=-1)
					# att_loss_op_ = tf.square(softmax_attention-distmap)*100 # attention softmax and distmap are between 0 and 1, 100 is for regularization
					att_loss_op_  = tf.square(softmax_attention[:,:,:,:,1]-distmap_1)*100
					att_loss_op_ = tf.expand_dims(att_loss_op_,axis=-1)
					att_loss_op = tf.reduce_mean(att_loss_op_)
				elif (FLAGS.attention_loss_function == "abs"):
					distmap_0 = 1. - tf.squeeze(distmap_placeholder,axis=-1)
					distmap_1 = tf.squeeze(distmap_placeholder,axis=-1)
					distmap = tf.stack([distmap_0,distmap_1],axis=-1)
					att_loss_op_ = tf.abs(softmax_attention-distmap)
					att_loss_op = tf.reduce_mean(att_loss_op_)
				else:
					sys.exit("Invalid loss function");
			tf.summary.scalar('attention_loss',att_loss_op)

			if FLAGS.image_log:
				for batch in range(FLAGS.batch_size):
					att_loss_log_0 = grayscale_to_rainbow(tf.transpose(att_loss_op_[batch:batch+1,:,:,:,0],[3,1,2,0]))
					# att_loss_log_1 = grayscale_to_rainbow(tf.transpose(att_loss_op_[batch:batch+1,:,:,:,1],[3,1,2,0]))
					att_loss_log_0 = tf.cast(tf.scalar_mul(255,att_loss_log_0), dtype=tf.uint8)
					# att_loss_log_1 = tf.cast(tf.scalar_mul(255,att_loss_log_1), dtype=tf.uint8)
				   
					# this two values is the same for binary classification
					tf.summary.image("att_loss_0", att_loss_log_0,max_outputs=FLAGS.patch_layer)
					# tf.summary.image("att_loss_1", att_loss_log_1,max_outputs=FLAGS.patch_layer)

			# total loss
			with tf.name_scope("total_loss"):
				total_loss_op = att_loss_op + loss_op
			tf.summary.scalar('total_loss',total_loss_op)

		# Argmax Op to generate label from logits
		with tf.name_scope("predicted_label"):
			pred_op = tf.argmax(logits_output, axis=4 , name="prediction")

		if FLAGS.image_log:
			for batch in range(FLAGS.batch_size):
				pred_log = tf.cast(tf.scalar_mul(255,pred_op[batch:batch+1,:,:,:]), dtype=tf.uint8)
				tf.summary.image("pred", tf.transpose(pred_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

		# Accuracy of model
		with tf.name_scope("accuracy"):
			correct_pred = tf.equal(tf.expand_dims(pred_op,-1), tf.cast(labels_placeholder,dtype=tf.int64))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		# metrics of the model
		with tf.name_scope("metrics"):
			tp, tp_op = tf.metrics.true_positives(labels_placeholder, pred_op, name="true_positives")
			tn, tn_op = tf.metrics.true_negatives(labels_placeholder, pred_op, name="true_negatives")
			fp, fp_op = tf.metrics.false_positives(labels_placeholder, pred_op, name="false_positives")
			fn, fn_op = tf.metrics.false_negatives(labels_placeholder, pred_op, name="false_negatives")
			sensitivity_op = tf.divide(tf.cast(tp_op,tf.float32),tf.cast(tf.add(tp_op,fn_op),tf.float32))
			specificity_op = tf.divide(tf.cast(tn_op,tf.float32),tf.cast(tf.add(tn_op,fp_op),tf.float32))
			dice_op = 2.*tp_op/(2.*tp_op+fp_op+fn_op)
		tf.summary.scalar('sensitivity', sensitivity_op)
		tf.summary.scalar('specificity', specificity_op)
		tf.summary.scalar('dice', dice_op)

		# Training Op
		with tf.name_scope("training"):
			# optimizer
			if FLAGS.optimizer == "sgd":
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.init_learning_rate)
			elif FLAGS.optimizer == "adam":
				optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.init_learning_rate)
			elif FLAGS.optimizer == "momentum":
				optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum)
			elif FLAGS.optimizer == "nesterov_momentum":
				optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum, use_nesterov=True)
			else:
				sys.exit("Invalid optimizer");

			if FLAGS.attention:
				train_op = optimizer.minimize(
					loss = total_loss_op,
					global_step=global_step)
			else:
				train_op = optimizer.minimize(
					loss = loss_op,
					global_step=global_step)

			# the update op is required by batch norm layer: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_ops])

		# # epoch checkpoint manipulation
		start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
		start_epoch_inc = start_epoch.assign(start_epoch+1)

		# saver
		summary_op = tf.summary.merge_all()
		checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir ,"checkpoint")
		print("Setting up Saver...")
		saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		# config.gpu_options.per_process_gpu_memory_fraction = 0.4

		# training cycle
		with tf.Session(config=config) as sess:
			# Initialize all variables
			sess.run(tf.global_variables_initializer())
			print("{}: Start training...".format(datetime.datetime.now()))

			# summary writer for tensorboard
			train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
			if FLAGS.testing:
				test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)

			# restore from checkpoint
			if FLAGS.restore_training:
				# check if checkpoint exists
				if os.path.exists(checkpoint_prefix+"-latest"):
					print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
					latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename="checkpoint-latest")
					saver.restore(sess, latest_checkpoint_path)
			
			print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()[0]))
			print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))

			# loop over epochs
			for epoch in np.arange(start_epoch.eval(), FLAGS.epochs):
				# initialize iterator in each new epoch
				sess.run(train_iterator.initializer)
				if FLAGS.testing:
					sess.run(test_iterator.initializer)
				print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

				# training phase
				while True:
					try:
						sess.run(tf.local_variables_initializer())
						if FLAGS.attention:
							[image, label, distMap] = sess.run(next_element_train)
							distMap = distMap[:,:,:,:,np.newaxis]
						else:
							[image, label] = sess.run(next_element_train)

						label = label[:,:,:,:,np.newaxis]
						
						model.is_training = True;
						if FLAGS.attention:
							train, summary, loss, att_loss = sess.run([train_op, summary_op, loss_op, att_loss_op], feed_dict={images_placeholder: image, labels_placeholder: label, distmap_placeholder: distMap, model.train_phase: True})
							print('{}: Training dice loss: {}'.format(datetime.datetime.now(), str(loss)))
							print('{}: Training attention loss: {}'.format(datetime.datetime.now(), str(att_loss)))
						else:
							train, summary, loss = sess.run([train_op, summary_op, loss_op], feed_dict={images_placeholder: image, labels_placeholder: label})
							print('{}: Training dice loss: {}'.format(datetime.datetime.now(), str(loss)))

						train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
						train_summary_writer.flush()

					except tf.errors.OutOfRangeError:
						start_epoch_inc.op.run()
						model.is_training = False;
						# print(start_epoch.eval())
						# save the model at end of each epoch training
						print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,FLAGS.checkpoint_dir))
						if not (os.path.exists(FLAGS.checkpoint_dir)):
							os.makedirs(FLAGS.checkpoint_dir,exist_ok=True)
						saver.save(sess, checkpoint_prefix, 
							global_step=tf.train.global_step(sess, global_step),
							latest_filename="checkpoint-latest")
						print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
						break
						
				if FLAGS.testing:
					# testing phase
					print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
					while True:
						try:
							sess.run(tf.local_variables_initializer())
							if FLAGS.attention:
								[image, label, distMap] = sess.run(next_element_test)
								distMap = distMap[:,:,:,:,np.newaxis]
							else:
								[image, label] = sess.run(next_element_test)

							label = label[:,:,:,:,np.newaxis]
							
							model.is_training = False;
							if FLAGS.attention:
								loss, summary, att_loss = sess.run([loss_op, summary_op, att_loss_op], feed_dict={images_placeholder: image, labels_placeholder: label, distmap_placeholder: distMap, model.train_phase: False})
							else:
								loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})

							test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
							train_summary_writer.flush()

						except tf.errors.OutOfRangeError:
							break

		# close tensorboard summary writer
		train_summary_writer.close()
		if FLAGS.testing:
			test_summary_writer.close()

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

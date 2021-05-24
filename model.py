import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from pipeline import NiftiDataset3D, NiftiDataset2D
import sys
import datetime
import numpy as np
import networks
import math
import SimpleITK as sitk
import multiprocessing
from tqdm import tqdm
import yaml

def grayscale_to_rainbow(image):
	# grayscale to rainbow colormap, convert to HSV (H = reversed grayscale from 0:2/3, S and V are all 1)
	# then convert to RGB
	H = tf.squeeze((1. - image)*2./3., axis=-1)
	SV = tf.ones(H.get_shape())
	HSV = tf.stack([H,SV,SV], axis=len(H.get_shape()))
	RGB = tf.image.hsv_to_rgb(HSV)

	return RGB

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), weights=[], smooth=1e-5):
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
	weight : list of float
		List of 1D batch-sized float-Tensors of the same length as chanel number.
	smooth : float
		This small value will be added to the numerator and denominator.
			- If both output and target are empty, it makes sure dice is 1.
			- If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

	Examples
	---------
	>>> import tensorlayer as tl
	>>> outputs = tl.act.pixel_wise_softmax(outputs)
	>>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

	References
	-----------
	- `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

	"""

	inse = tf.reduce_sum(output * target, axis=axis)
	if loss_type == 'jaccard':
		l = tf.reduce_sum(output * output, axis=axis)
		r = tf.reduce_sum(target * target, axis=axis)
	elif loss_type == 'sorensen':
		l = tf.reduce_sum(output, axis=axis)
		r = tf.reduce_sum(target, axis=axis)
	else:
		raise Exception("Unknown loss_type")

	if weights != []:
		assert len(weights) == target.get_shape()[-1], "Length of DICE weight is {}, should be {}".format(len(weights),target.get_shape()[-1])
		weights = tf.cast(weights,tf.float32)
		w = 1./(tf.reduce_sum(target*target, axis=axis) + smooth)
		dice = tf.reduce_sum(2.* weights * inse + smooth, axis=-1)/tf.reduce_sum(weights*(l + r) + smooth,axis=-1)
		dice = tf.reduce_mean(dice, name='dice_coe')
	else:
		# old axis=[0,1,2,3]
		# dice = 2 * (inse) / (l + r)
		# epsilon = 1e-5
		# dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
		# new haodong
		dice = (2. * inse + smooth) / (l + r + smooth)
		dice = tf.reduce_mean(dice, name='dice_coe')

	return dice

def weighted_softmax_cross_entropy_with_logits(labels,logits, weights):
	class_weights = tf.constant([weights])
	weights = tf.reduce_sum(class_weights * labels, axis=-1)
	unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	weighted_losses = unweighted_losses * weights
	return tf.reduce_mean(weighted_losses)

def prepare_batch(image_ijk_patch_indices_dict):
	# image_batches = []
	# for batch in ijk_patch_indices:
	# 	image_batch = []
	# 	for patch in batch:
	# 		image_patch = images[patch[0]:patch[1],patch[2]:patch[3],patch[4]:patch[5],:]
	# 		image_batch.append(image_patch)

	# 	image_batch = np.asarray(image_batch)
	# 	image_batches.append(image_batch)
	
	images, ijk_patch_indices = image_ijk_patch_indices_dict['images'], image_ijk_patch_indices_dict['indexes']

	# return image_batches
	image_batch = []
	for patch in ijk_patch_indices:
		image_patch = images[patch[0]:patch[1],patch[2]:patch[3],patch[4]:patch[5],:]
		image_batch.append(image_patch)

	image_batch = np.asarray(image_batch)
		
	return image_batch

def volume_threshold(image,volume):
	ccFilter = sitk.ConnectedComponentImageFilter()
	image = ccFilter.Execute(image)

	statFilter = sitk.LabelShapeStatisticsImageFilter()
	statFilter.Execute(image)

	output_image = sitk.Image(image.GetSize(),sitk.sitkUInt8)
	output_image.SetOrigin(image.GetOrigin())
	output_image.SetSpacing(image.GetSpacing())
	output_image.SetDirection(image.GetDirection())

	for label in statFilter.GetLabels():
		if statFilter.GetPhysicalSize(label)> volume:
			thresholdFilter = sitk.BinaryThresholdImageFilter()
			thresholdFilter.SetLowerThreshold(label)
			thresholdFilter.SetUpperThreshold(label)
			thresholdFilter.SetInsideValue(1)
			thres_image = thresholdFilter.Execute(image)

			addFilter = sitk.AddImageFilter()
			output_image = addFilter.Execute(output_image,thres_image)

	return output_image

def ExtractLargestConnectedComponents(label):
	castFilter = sitk.CastImageFilter()
	castFilter.SetOutputPixelType(sitk.sitkUInt8)
	label = castFilter.Execute(label)

	ccFilter = sitk.ConnectedComponentImageFilter()
	label = ccFilter.Execute(label)

	labelStat = sitk.LabelShapeStatisticsImageFilter()
	labelStat.Execute(label)

	largestVol = 0
	largestLabel = 0
	for labelNum in labelStat.GetLabels():
		if labelStat.GetPhysicalSize(labelNum) > largestVol:
			largestVol = labelStat.GetPhysicalSize(labelNum)
			largestLabel = labelNum
	
	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetLowerThreshold(largestLabel)
	thresholdFilter.SetUpperThreshold(largestLabel)
	thresholdFilter.SetInsideValue(1)
	thresholdFilter.SetOutsideValue(0)
	label = thresholdFilter.Execute(label)

	return label

class image2label(object):
	def __init__(self,sess,config):
		"""
		Args:
			sess: Tensorflow session
			config: Model configuration
		"""
		self.sess = sess
		self.config = config
		self.model = None
		self.graph = tf.Graph()
		self.graph.as_default()
		self.epoches = 999999999999999999

	def read_config(self):
		print("{}: Reading configuration file...".format(datetime.datetime.now()))

		# training config
		self.input_channel_num = len(self.config['TrainingSetting']['Data']['ImageFilenames'])
		self.output_channel_num = len(self.config['TrainingSetting']['SegmentationClasses'])
		self.label_classes = self.config['TrainingSetting']['SegmentationClasses']

		self.train_data_dir = self.config['TrainingSetting']['Data']['TrainingDataDirectory']
		self.test_data_dir = self.config['TrainingSetting']['Data']['TestingDataDirectory']
		self.image_filenames = self.config['TrainingSetting']['Data']['ImageFilenames']
		self.label_filename = self.config['TrainingSetting']['Data']['LabelFilename']

		self.batch_size = self.config['TrainingSetting']['BatchSize']
		self.patch_shape = self.config['TrainingSetting']['PatchShape']
		self.dimension = len(self.config['TrainingSetting']['PatchShape'])
		self.image_log = self.config['TrainingSetting']['ImageLog']
		self.testing = self.config['TrainingSetting']['Testing']
		self.test_step = self.config['TrainingSetting']['TestStep']

		self.restore_training = self.config['TrainingSetting']['Restore']
		self.log_dir = self.config['TrainingSetting']['LogDir']
		self.ckpt_dir = self.config['TrainingSetting']['CheckpointDir']
		self.epoches = self.config['TrainingSetting']['Epoches']
		self.max_itr = self.config['TrainingSetting']['MaxIterations']
		self.log_interval = self.config['TrainingSetting']['LogInterval']

		self.network_name = self.config['TrainingSetting']['Networks']['Name']
		self.dropout_rate = self.config['TrainingSetting']['Networks']['Dropout']

		self.optimizer_name = self.config['TrainingSetting']['Optimizer']['Name']
		self.initial_learning_rate = self.config['TrainingSetting']['Optimizer']['InitialLearningRate']
		self.decay_factor = self.config['TrainingSetting']['Optimizer']['Decay']['Factor']
		self.decay_steps = self.config['TrainingSetting']['Optimizer']['Decay']['Steps']
		self.spacing = self.config['TrainingSetting']['Spacing']
		self.drop_ratio = self.config['TrainingSetting']['DropRatio']
		self.min_pixel = self.config['TrainingSetting']['MinPixel']

		self.loss_name = self.config['TrainingSetting']['Loss']['Name']
		self.loss_weights = self.config['TrainingSetting']['Loss']['Weights']
		self.loss_alpha = self.config['TrainingSetting']['Loss']['Alpha']
		self.training_pipeline = self.config['TrainingSetting']['Pipeline']

		# evaluation config
		self.checkpoint_path = self.config['EvaluationSetting']['CheckpointPath']
		self.evaluate_data_dir = self.config['EvaluationSetting']['Data']['EvaluateDataDirectory']
		self.evaluate_image_filenames = self.config['EvaluationSetting']['Data']['ImageFilenames']
		self.evaluate_label_filename = self.config['EvaluationSetting']['Data']['LabelFilename']
		self.evaluate_probability_filename = self.config['EvaluationSetting']['Data']['ProbabilityFilename']
		self.evaluate_stride = self.config['EvaluationSetting']['Stride']
		self.evaluate_batch = self.config['EvaluationSetting']['BatchSize']
		self.evaluate_probability_output = self.config['EvaluationSetting']['ProbabilityOutput']
		self.evaluate_lcc = self.config['EvaluationSetting']['LargestConnectedComponent']
		self.evaluate_volume_threshold = self.config['EvaluationSetting']['VolumeThreshold']
		self.evaluate_pipeline = self.config['EvaluationSetting']['Pipeline']

		print("{}: Reading configuration file complete".format(datetime.datetime.now()))

	def placeholder_inputs(self, input_batch_shape, output_batch_shape):
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

	def dataset_iterator(self, data_dir, transforms, train=True):
		if self.dimension==2:
			Dataset = NiftiDataset2D.NiftiDataset(
					data_dir=data_dir,
					image_filenames=self.image_filenames,
					label_filename=self.label_filename,
					transforms3D=transforms['3D'],
					transforms2D=transforms['2D'],
					train=train,
					labels=self.label_classes
				)
		else:
			Dataset = NiftiDataset3D.NiftiDataset(
				data_dir=data_dir,
				image_filenames=self.image_filenames,
				label_filename=self.label_filename,
				transforms=transforms,
				train=train,
				labels=self.label_classes
			)
		
		dataset = Dataset.get_dataset()
		if self.dimension == 2:
			dataset = dataset.shuffle(buffer_size=5)
		else:
			dataset = dataset.shuffle(buffer_size=3)
		dataset = dataset.batch(self.batch_size,drop_remainder=True)

		return dataset.make_initializable_iterator()

	def build_model_graph(self):
		print("{}: Start to build model graph...".format(datetime.datetime.now()))

		self.global_step_op = tf.train.get_or_create_global_step()		

		if self.dimension == 2:
			input_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.input_channel_num) 
			output_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], 1) 
		elif self.dimension == 3:
			input_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], self.input_channel_num) 
			output_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], 1) 
		else:
			sys.exit('Invalid Patch Shape (length should be 2 or 3)')

		self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(input_batch_shape,output_batch_shape)
		self.dropout_placeholder = tf.placeholder(tf.float32,name="dropout_placeholder")

		# plot input and output images to tensorboard
		if self.image_log:
			if self.dimension == 2:
				for image_channel in range(self.input_channel_num):
					image_log = tf.cast(self.images_placeholder[:,:,:,image_channel:image_channel+1], dtype=tf.uint8)
					tf.summary.image(self.image_filenames[image_channel], image_log, max_outputs=self.batch_size)
				if 0 in self.label_classes:
					labels_log = tf.cast(self.labels_placeholder*math.floor(255/(self.output_channel_num-1)), dtype=tf.uint8)
				else:
					labels_log = tf.cast(self.labels_placeholder*math.floor(255/self.output_channel_num), dtype=tf.uint8)
				tf.summary.image("label",labels_log, max_outputs=self.batch_size)
			else:
				for batch in range(self.batch_size):
					for image_channel in range(self.input_channel_num):
						image_log = tf.cast(self.images_placeholder[batch:batch+1,:,:,:,image_channel], dtype=tf.uint8)
						tf.summary.image(self.image_filenames[image_channel]+"_batch"+str(batch), tf.transpose(image_log,[3,1,2,0]),max_outputs=self.patch_shape[-1])
					if 0 in self.label_classes:
						labels_log = tf.cast(self.labels_placeholder[batch:batch+1,:,:,:,0]*math.floor(255/(self.output_channel_num-1)),dtype=tf.uint8)
					else:
						labels_log = tf.cast(self.labels_placeholder[batch:batch+1,:,:,:,0]*math.floor(255/self.output_channel_num), dtype=tf.uint8)
					tf.summary.image("label"+"_batch"+str(batch), tf.transpose(labels_log,[3,1,2,0]),max_outputs=self.patch_shape[-1])

		# Get images and labels
		# create transformations to image and labels
		# Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
		with tf.device('/cpu:0'):
			# load the pipeline from yaml
			with open(self.training_pipeline) as f:
				pipeline_ = yaml.load(f)

			if self.dimension == 2:
				train_transforms_3d = []
				train_transforms_2d = []
				test_transforms_3d = []
				test_transforms_2d = []

				if pipeline_["preprocess"]["train"]["3D"] is not None:
					for transform in pipeline_["preprocess"]["train"]["3D"]:
						try:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
						except:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])()
						train_transforms_3d.append(tfm_cls)

				if pipeline_["preprocess"]["train"]["2D"] is not None:
					for transform in pipeline_["preprocess"]["train"]["2D"]:
						try:
							tfm_cls = getattr(NiftiDataset2D,transform["name"])(*[],**transform["variables"])
						except:
							tfm_cls = getattr(NiftiDataset2D,transform["name"])()
						train_transforms_2d.append(tfm_cls)

				if pipeline_["preprocess"]["test"]["3D"] is not None:
					for transform in pipeline_["preprocess"]["test"]["3D"]:
						try:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
						except:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])()
						test_transforms_3d.append(tfm_cls)

				if pipeline_["preprocess"]["test"]["2D"] is not None:
					for transform in pipeline_["preprocess"]["test"]["2D"]:
						try:
							tfm_cls = getattr(NiftiDataset2D,transform["name"])(*[],**transform["variables"])
						except:
							tfm_cls = getattr(NiftiDataset2D,transform["name"])()
						test_transforms_2d.append(tfm_cls)

				trainTransforms = {"3D": train_transforms_3d, "2D": train_transforms_2d}
				testTransforms = {"3D": test_transforms_3d, "2D": test_transforms_2d}
			else:
				trainTransforms = []
				testTransforms = []

				if pipeline_["preprocess"]["train"]["3D"] is not None:
					for transform in pipeline_["preprocess"]["train"]["3D"]:
						try:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
						except KeyError:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])()
						trainTransforms.append(tfm_cls)

				if pipeline_["preprocess"]["test"]["3D"] is not None:
					for transform in pipeline_["preprocess"]["test"]["3D"]:
						try:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
						except KeyError:
							tfm_cls = getattr(NiftiDataset3D,transform["name"])()
						testTransforms.append(tfm_cls)

			# get input and output datasets
			self.train_iterator = self.dataset_iterator(self.train_data_dir, trainTransforms)
			self.next_element_train = self.train_iterator.get_next()

			if self.testing:
				self.test_iterator = self.dataset_iterator(self.test_data_dir, testTransforms)
				self.next_element_test = self.test_iterator.get_next()

		print("{}: Dataset pipeline complete".format(datetime.datetime.now()))

		# network models:
		if self.network_name == "FCN":
			sys.exit("Network to be developed")
		elif self.network_name == "UNet":
			self.network = networks.UNet(
				num_output_channels=self.output_channel_num,
				dropout_rate=self.dropout_placeholder,
				num_channels=4,
				num_levels=4,
				num_convolutions=2,
				bottom_convolutions=2,
				is_training=True,
				activation_fn="relu"
				)
		elif self.network_name =="VNet":
			self.network = networks.VNet(
				num_classes=self.output_channel_num,
				dropout_rate=self.dropout_placeholder,
				num_channels=16,
				num_levels=4,
				num_convolutions=(1, 2, 3, 3),
				bottom_convolutions=3,
				is_training = True,
				activation_fn="prelu"
				)
		else:
			sys.exit("Invalid Network")

		print("{}: Core network complete".format(datetime.datetime.now()))

		self.logits = self.network.GetNetwork(self.images_placeholder)

		# softmax op
		self.softmax_op = tf.nn.softmax(self.logits,name="softmax")

		if self.image_log:
			if self.dimension == 2:
				for output_channel in range(self.output_channel_num):
					softmax_log = []
					for batch in range(self.batch_size):
						softmax_log.append(grayscale_to_rainbow(self.softmax_op[batch,:,:,output_channel:output_channel+1]))
					softmax_log = tf.stack(softmax_log,axis=0)
					softmax_log = tf.cast(softmax_log*255, dtype = tf.uint8)
					tf.summary.image("softmax_" + str(self.label_classes[output_channel]),softmax_log,max_outputs=self.batch_size)
			else:
				for batch in range(self.batch_size):
					for output_channel in range(self.output_channel_num):
						softmax_log = grayscale_to_rainbow(tf.transpose(self.softmax_op[batch:batch+1,:,:,:,output_channel],[3,1,2,0]))
						softmax_log = tf.cast(softmax_log*255,dtype=tf.uint8)
						tf.summary.image("softmax_" + str(self.label_classes[output_channel])+"_batch"+str(batch),softmax_log,max_outputs=self.patch_shape[-1])

		print("{}: Output layers complete".format(datetime.datetime.now()))

		# loss function
		with tf.name_scope("loss"):
			# """
			# 	Tricks for faster converge: Here we provide two calculation methods, first one will ignore  to classical dice formula
			# 	method 1: exclude the 0-th label in dice calculation. to use this method properly, you must set 0 as the first value in SegmentationClasses in config.json
			# 	method 2: dice will be average on all classes
			# """
			if self.dimension == 2:
				labels = tf.one_hot(self.labels_placeholder[:,:,:,0], depth=self.output_channel_num)
			else:
				labels = tf.one_hot(self.labels_placeholder[:,:,:,:,0], depth=self.output_channel_num)

			# if 0 in self.label_classes:
			# 	################### method 1 ###################
			# 	if self.dimension ==2:
			# 		labels = labels[:,:,:,1:]
			# 		softmax = self.softmax_op[:,:,:,1:]
			# 	else:
			# 		labels = labels[:,:,:,:,1:]
			# 		softmax = self.softmax_op[:,:,:,:,1:]
			# else:
			# 	################### method 2 ###################
			# 	labels = labels
			# 	softmax = self.softmax_op

			labels = labels
			softmax = self.softmax_op

			if (self.loss_name == "xent"):
				self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=self.logits))
			if (self.loss_name == "weighted_xent"):
				self.loss_op = weighted_softmax_cross_entropy_with_logits(labels,self.logits,self.loss_weights)
			elif (self.loss_name == "sorensen"):
				if self.dimension == 2:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen',axis=(1,2))
				else:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen')
				self.loss_op = 1. - sorensen
			elif (self.loss_name == "weighted_sorensen"):
				if self.dimension == 2:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen', axis=(1,2), weights=self.loss_weights)
				else:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen', weights=self.loss_weights)
				self.loss_op = 1. - sorensen
			elif (self.loss_name == "jaccard"):
				if self.dimension == 2:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard',axis=(1,2))
				else:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard')
				self.loss_op = 1. - jaccard
			elif (self.loss_name == "weighted_jaccard"):
				if self.dimension == 2:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard',axis=(1,2), weights=self.loss_weights)
				else:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard', weights=self.loss_weights)
				self.loss_op = 1. - jaccard
			elif (self.loss_name == "mixed_sorensen"):
				xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=self.logits))
				if self.dimension == 2:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen',axis=(1,2))
				else:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen')
				tf.summary.scalar('1.dice', (1. - sorensen))
				tf.summary.scalar('2.regularized_xent', self.loss_alpha*xent)
				self.loss_op = (1. - sorensen) + self.loss_alpha*xent
			elif (self.loss_name == "mixed_weighted_sorensen"):
				xent = weighted_softmax_cross_entropy_with_logits(labels,self.logits,self.loss_weights)
				if self.dimension == 2:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen', axis=(1,2), weights=self.loss_weights)
				else:
					sorensen = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='sorensen', weights=self.loss_weights)
				tf.summary.scalar('1.dice', (1. - sorensen))
				tf.summary.scalar('2.regularized_xent', self.loss_alpha*xent)
				self.loss_op = (1. - sorensen) + self.loss_alpha*xent
			elif (self.loss_name == "mixed_jaccard"):
				xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=self.logits))
				if self.dimension == 2:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard',axis=(1,2))
				else:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard')
				tf.summary.scalar('1.dice', (1. - jaccard))
				tf.summary.scalar('2.regularized_xent', self.loss_alpha*xent)
				self.loss_op = (1. - jaccard) + self.loss_alpha*xent
			elif (self.loss_name == "mixed_weighted_jaccard"):
				xent = weighted_softmax_cross_entropy_with_logits(labels,self.logits,self.loss_weights)
				if self.dimension == 2:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard',axis=(1,2), weights=self.loss_weights)
				else:
					jaccard = dice_coe(softmax,tf.cast(labels,dtype=tf.float32), loss_type='jaccard', weights=self.loss_weights)
				tf.summary.scalar('1.dice', (1. - jaccard))
				tf.summary.scalar('2.regularized_xent', self.loss_alpha*xent)
				self.loss_op = (1. - jaccard) + self.loss_alpha*xent
			else:
				sys.exit("Invalid loss function")

			tf.summary.scalar('0.total_loss', self.loss_op)

		print("{}: Loss function complete".format(datetime.datetime.now()))

		# argmax op
		with tf.name_scope("predicted_label"):
			self.pred_op = tf.argmax(self.logits, axis=-1 , name="prediction")

		if self.image_log:
			if self.dimension == 2:
				if 0 in self.label_classes:
					pred_log = tf.cast(self.pred_op*math.floor(255/(self.output_channel_num-1)),dtype=tf.uint8)
				else:
					pred_log = tf.cast(self.pred_op*math.floor(255/self.output_channel_num),dtype=tf.uint8)
				pred_log = tf.expand_dims(pred_log,axis=-1)
				tf.summary.image("pred", pred_log, max_outputs=self.batch_size)
			else:
				for batch in range(self.batch_size):
					if 0 in self.label_classes:
						pred_log = tf.cast(self.pred_op[batch:batch+1,:,:,:]*math.floor(255/(self.output_channel_num-1)), dtype=tf.uint8)
					else:
						pred_log = tf.cast(self.pred_op[batch:batch+1,:,:,:]*math.floor(255/(self.output_channel_num)), dtype=tf.uint8)
					
					tf.summary.image("pred"+"_batch"+str(batch), tf.transpose(pred_log,[3,1,2,0]),max_outputs=self.patch_shape[-1])

		# accuracy of the model
		with tf.name_scope("metrics"):
			correct_pred = tf.equal(tf.expand_dims(self.pred_op,-1), tf.cast(self.labels_placeholder,dtype=tf.int64))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

			tf.summary.scalar('accuracy', accuracy)

			# confusion matrix
			if self.dimension == 2:
				label_one_hot = tf.one_hot(self.labels_placeholder[:,:,:,0], depth=self.output_channel_num)
				pred_one_hot = tf.one_hot(self.pred_op, depth=self.output_channel_num)
			else:
				label_one_hot = tf.one_hot(self.labels_placeholder[:,:,:,:,0],depth=self.output_channel_num)
				pred_one_hot = tf.one_hot(self.pred_op[:,:,:,:], depth=self.output_channel_num)

			for i in range(self.output_channel_num):
				if i == 0:
					continue
				else:
					if self.dimension == 2:
						tp, tp_op = tf.metrics.true_positives(label_one_hot[:,:,:,i], pred_one_hot[:,:,:,i], name="true_positives_"+str(self.label_classes[i]))
						tn, tn_op = tf.metrics.true_negatives(label_one_hot[:,:,:,i], pred_one_hot[:,:,:,i], name="true_negatives_"+str(self.label_classes[i]))
						fp, fp_op = tf.metrics.false_positives(label_one_hot[:,:,:,i], pred_one_hot[:,:,:,i], name="false_positives_"+str(self.label_classes[i]))
						fn, fn_op = tf.metrics.false_negatives(label_one_hot[:,:,:,i], pred_one_hot[:,:,:,i], name="false_negatives_"+str(self.label_classes[i]))
						auc, auc_op = tf.metrics.auc(label_one_hot[:,:,:,i], self.softmax_op[:,:,:,i],name="auc_"+str(self.label_classes[i]))
					else:
						tp, tp_op = tf.metrics.true_positives(label_one_hot[:,:,:,:,i], pred_one_hot[:,:,:,:,i], name="true_positives_"+str(self.label_classes[i]))
						tn, tn_op = tf.metrics.true_negatives(label_one_hot[:,:,:,:,i], pred_one_hot[:,:,:,:,i], name="true_negatives_"+str(self.label_classes[i]))
						fp, fp_op = tf.metrics.false_positives(label_one_hot[:,:,:,:,i], pred_one_hot[:,:,:,:,i], name="false_positives_"+str(self.label_classes[i]))
						fn, fn_op = tf.metrics.false_negatives(label_one_hot[:,:,:,:,i], pred_one_hot[:,:,:,:,i], name="false_negatives_"+str(self.label_classes[i]))
						auc, auc_op = tf.metrics.auc(label_one_hot[:,:,:,:,i], self.softmax_op[:,:,:,:,i],name="auc_"+str(self.label_classes[i]))

					sensitivity_op = tf.divide(tf.cast(tp_op,tf.float32),tf.cast(tf.add(tp_op,fn_op),tf.float32))
					specificity_op = tf.divide(tf.cast(tn_op,tf.float32),tf.cast(tf.add(tn_op,fp_op),tf.float32))
					dice_op = 2.*tp_op/(2.*tp_op+fp_op+fn_op)

				tf.summary.scalar('sensitivity_'+str(self.label_classes[i]), sensitivity_op)
				tf.summary.scalar('specificity_'+str(self.label_classes[i]), specificity_op)
				tf.summary.scalar('dice_'+str(self.label_classes[i]), dice_op)
				tf.summary.scalar('auc_'+str(self.label_classes[i]), auc_op)

		print("{}: Metrics complete".format(datetime.datetime.now()))

		print("{}: Build graph complete".format(datetime.datetime.now()))

	def train(self):
		# read config to class variables
		self.read_config()

		"""Train image2label model"""
		self.build_model_graph()

		# learning rate
		with tf.name_scope("learning_rate"):
			learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step_op,
				self.decay_steps,self.decay_factor,staircase=False)
		tf.summary.scalar('learning_rate', learning_rate)

		# optimizer
		with tf.name_scope("training"):
			# optimizer
			if self.optimizer_name == "SGD":
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			elif self.optimizer_name == "Adam":
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			elif self.optimizer_name == "Momentum":
				optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum)
			elif self.optimizer_name == "NesterovMomentum":
				optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum, use_nesterov=True)
			else:
				sys.exit("Invalid optimizer");

			train_op = optimizer.minimize(
				loss = self.loss_op,
				global_step=self.global_step_op)

			# the update op is required by batch norm layer: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group([train_op, update_ops])

		start_epoch = tf.get_variable("start_epoch", shape=[1], initializer=tf.zeros_initializer, dtype=tf.int32)
		start_epoch_inc = start_epoch.assign(start_epoch+1)

		# actual training cycle
		# initialize all variables
		self.sess.run(tf.initializers.global_variables())
		print("{}: Start training...".format(datetime.datetime.now()))

		# saver
		print("{}: Setting up Saver...".format(datetime.datetime.now()))
		if not self.restore_training:
			# clear log directory
			if os.path.exists(self.log_dir):
				shutil.rmtree(self.log_dir)
			os.makedirs(self.log_dir)

			# clear checkpoint directory
			if os.path.exists(self.ckpt_dir):
				shutil.rmtree(self.ckpt_dir)
			os.makedirs(self.ckpt_dir)

			saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
			checkpoint_prefix = os.path.join(self.ckpt_dir,"checkpoint")
		else:
			saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
			checkpoint_prefix = os.path.join(self.ckpt_dir,"checkpoint")

			# check if checkpoint exists
			if os.path.exists(checkpoint_prefix+"-latest"):
				print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),self.ckpt_dir))
				latest_checkpoint_path = tf.train.latest_checkpoint(self.ckpt_dir,latest_filename="checkpoint-latest")
				saver.restore(self.sess, latest_checkpoint_path)
			
			print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval(session=self.sess)[0]))
			print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(self.sess, self.global_step_op)))

		summary_op = tf.summary.merge_all()

		# summary writer for tensorboard
		train_summary_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
		if self.testing:
			test_summary_writer = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)

		# testing initializer need to execute outside training loop
		if self.testing:
			self.sess.run(self.test_iterator.initializer)

		# loop over epochs
		for epoch in np.arange(start_epoch.eval(session=self.sess), self.epoches):
			print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch+1))
			# initialize iterator in each new epoch
			self.sess.run(self.train_iterator.initializer)
			
			# print("{}: Dataset iterator initialize ok".format(datetime.datetime.now()))

			# training phase
			loss_sum = 0
			count = 0
			while True:
				if self.global_step_op.eval() > self.max_itr:
					sys.exit("{}: Reach maximum iteration steps, training abort.".format(datetime.datetime.now()))
				try:
					self.sess.run(tf.local_variables_initializer())

					# print("{}: Local variable initialize ok".format(datetime.datetime.now()))
					# self.network.is_training = True
					print("{}: Set network to training ok".format(datetime.datetime.now()))
					image, label = self.sess.run(self.next_element_train)
					print("{}: Get next element train ok".format(datetime.datetime.now()))

					if self.dimension == 2:
						label = label[:,:,:,np.newaxis]
					else:
						label = label[:,:,:,:,np.newaxis]

					train, summary, loss = self.sess.run([train_op,summary_op,self.loss_op], feed_dict={
						self.images_placeholder: image,
						self.labels_placeholder: label,
						self.dropout_placeholder: self.dropout_rate,
						self.network.train_phase: True
						})
					print('{}: Segmentation training loss: {}'.format(datetime.datetime.now(), str(loss)))

					loss_sum += loss
					count += 1

					train_summary_writer.add_summary(summary,global_step=tf.train.global_step(self.sess,self.global_step_op))
					train_summary_writer.flush()

					# save checkpoint
					if self.global_step_op.eval()%self.log_interval == 0:
						print("{}: Saving checkpoint of step {} at {}...".format(datetime.datetime.now(),self.global_step_op.eval(),self.ckpt_dir))
						if not (os.path.exists(self.ckpt_dir)):
							os.makedirs(self.ckpt_dir,exist_ok=True)
						saver.save(self.sess, checkpoint_prefix, 
							global_step=tf.train.global_step(self.sess, self.global_step_op),
							latest_filename="checkpoint-latest")

					# testing phase
					if self.testing and (self.global_step_op.eval()%self.test_step == 0):
						self.sess.run(tf.local_variables_initializer())
						print("{}: Set network to training ok".format(datetime.datetime.now()))
						# train_phase = False
						# self.network.is_training = train_phase
						try:
							image, label = self.sess.run(self.next_element_test)
						except tf.errors.OutOfRangeError:
							self.sess.run(self.test_iterator.initializer)
							image, label = self.sess.run(self.next_element_test)
						print("{}: Get next element test ok".format(datetime.datetime.now()))
								
						if self.dimension == 2:
							label = label[:,:,:,np.newaxis]
						else:
							label = label[:,:,:,:,np.newaxis]

						summary, loss = self.sess.run([summary_op, self.loss_op],feed_dict={
							self.images_placeholder: image,
							self.labels_placeholder: label,
							self.dropout_placeholder: 0.0,
							self.network.train_phase: True
						})

						print('{}: Segmentation testing loss: {}'.format(datetime.datetime.now(), str(loss)))

						test_summary_writer.add_summary(summary, global_step=tf.train.global_step(self.sess, self.global_step_op))
						test_summary_writer.flush()

				except tf.errors.OutOfRangeError:
					print("{}: Training of epoch {} complete, epoch loss: {}".format(datetime.datetime.now(),epoch+1,loss_sum/count))

					start_epoch_inc.op.run()
					# self.network.is_training = False;
					# print(start_epoch.eval())
					# save the model at end of each epoch training
					print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,self.ckpt_dir))
					if not (os.path.exists(self.ckpt_dir)):
						os.makedirs(self.ckpt_dir,exist_ok=True)
					saver.save(self.sess, checkpoint_prefix, 
						global_step=tf.train.global_step(self.sess, self.global_step_op),
						latest_filename="checkpoint-latest")
					print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
					break
				
		# close tensorboard summary writer
		train_summary_writer.close()
		if self.testing:
			test_summary_writer.close()

	def evaluate_single_3D(self, sample, transforms):
		input_origin = sample['image'][0].GetOrigin()
		input_direction = sample['image'][0].GetDirection()
		input_spacing = sample['image'][0].GetSpacing()
		input_size = sample['image'][0].GetSize()

		for transform in transforms:
			sample = transform(sample)

		images = sample['image']
		label = sample['label']

		softmax_tfm = []
		for channel in range(self.output_channel_num):
			# create empty softmax image in pair with transformed image
			softmax_tfm_ = sitk.Image(images[0].GetSize(),sitk.sitkFloat32)
			softmax_tfm_.SetOrigin(images[0].GetOrigin())
			softmax_tfm_.SetDirection(images[0].GetDirection())
			softmax_tfm_.SetSpacing(images[0].GetSpacing())
			softmax_tfm.append(softmax_tfm_)

		# convert image to numpy array
		for channel in range(self.input_channel_num):
			image_ = sitk.GetArrayFromImage(images[channel])
			if channel == 0:
				images_np = image_[:,:,:,np.newaxis]
			else:
				images_np = np.append(images_np, image_[:,:,:,np.newaxis], axis=-1)

		images_np = np.asarray(images_np,np.float32)
		label_np = sitk.GetArrayFromImage(label)
		label_np = np.asarray(label_np,np.int32)

		softmax_np = []
		for channel in range(self.output_channel_num):
			softmax_np_ = sitk.GetArrayFromImage(softmax_tfm[channel])
			softmax_np_ = np.asarray(softmax_np_,np.float32)
			softmax_np.append(softmax_np_)

		# unify numpy and sitk orientation
		images_np = np.transpose(images_np,(2,1,0,3))
		label_np = np.transpose(label_np,(2,1,0))
		for channel in range(self.output_channel_num):
			softmax_np[channel] = np.transpose(softmax_np[channel],(2,1,0))

		# a weighting matrix will be used for averaging the overlapped region
		weight_np = np.zeros(label_np.shape)

		# prepare image batch indices
		inum = int(math.ceil((images_np.shape[0]-self.patch_shape[0])/float(self.evaluate_stride[0]))) + 1 
		jnum = int(math.ceil((images_np.shape[1]-self.patch_shape[1])/float(self.evaluate_stride[1]))) + 1
		knum = int(math.ceil((images_np.shape[2]-self.patch_shape[2])/float(self.evaluate_stride[2]))) + 1

		patch_total = 0
		image_ijk_patch_indices_dicts = []
		ijk_patch_indicies_tmp = []

		for i in range(inum):
			for j in range(jnum):
				for k in range (knum):
					if patch_total % self.evaluate_batch == 0:
						ijk_patch_indicies_tmp = []

					istart = i * self.evaluate_stride[0]
					if istart + self.patch_shape[0] > images_np.shape[0]: #for last patch
						istart = images_np.shape[0] - self.patch_shape[0]
					iend = istart + self.patch_shape[0]

					jstart = j * self.evaluate_stride[1]
					if jstart + self.patch_shape[1] > images_np.shape[1]: #for last patch
						jstart = images_np.shape[1] - self.patch_shape[1]
					jend = jstart + self.patch_shape[1]

					kstart = k * self.evaluate_stride[2]
					if kstart + self.patch_shape[2] > images_np.shape[2]: #for last patch
						kstart = images_np.shape[2] - self.patch_shape[2] 
					kend = kstart + self.patch_shape[2]

					ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

					if patch_total % self.evaluate_batch == 0:
						image_ijk_patch_indices_dicts.append({'images': images_np, 'indexes':ijk_patch_indicies_tmp})

					patch_total += 1

		# for last batch
		image_ijk_patch_indices_dicts.append({'images': images_np, 'indexes':ijk_patch_indicies_tmp})

		p = multiprocessing.Pool(multiprocessing.cpu_count())
		batches = p.map(prepare_batch,image_ijk_patch_indices_dicts)
		p.close()
		p.join()

		# actual segmentation
		for i in tqdm(range(len(batches))):
			batch = batches[i]

			[pred, softmax] = self.sess.run(['predicted_label/prediction:0','softmax:0'], feed_dict={
				'images_placeholder:0': batch, 
				'dropout_placeholder:0': 0.0,
				'train_phase_placeholder:0': True})

			for j in range(pred.shape[0]):
				istart = image_ijk_patch_indices_dicts[i]['indexes'][j][0]
				iend = image_ijk_patch_indices_dicts[i]['indexes'][j][1]
				jstart = image_ijk_patch_indices_dicts[i]['indexes'][j][2]
				jend = image_ijk_patch_indices_dicts[i]['indexes'][j][3]
				kstart = image_ijk_patch_indices_dicts[i]['indexes'][j][4]
				kend = image_ijk_patch_indices_dicts[i]['indexes'][j][5]

				for channel in range(self.output_channel_num):
					softmax_np[channel][istart:iend,jstart:jend,kstart:kend] += softmax[j,:,:,:,channel]
				weight_np[istart:iend,jstart:jend,kstart:kend] += 1.0

		print("{}: Evaluation complete".format(datetime.datetime.now()))
		# eliminate overlapping region using the weighted value
		# label_np = np.rint(np.float32(label_np)/np.float32(weight_np) + 0.01)
		label_np = np.argmax(softmax_np,axis=0)
		if self.evaluate_probability_output:
			for channel in range(self.output_channel_num):
				softmax_np[channel] = softmax_np[channel]/np.float32(weight_np)

		# convert back to sitk space
		label_np = np.transpose(label_np,(2,1,0))
		if self.evaluate_probability_output:
			for channel in range(self.output_channel_num):
				softmax_np[channel] = np.transpose(softmax_np[channel],(2,1,0))

		# convert label numpy back to sitk image
		label_tfm = sitk.GetImageFromArray(label_np)
		label_tfm.SetOrigin(images[0].GetOrigin())
		label_tfm.SetDirection(images[0].GetDirection())
		label_tfm.SetSpacing(images[0].GetSpacing())

		for channel in range(self.output_channel_num):
			softmax_tfm[channel] = sitk.GetImageFromArray(softmax_np[channel])
			softmax_tfm[channel].SetOrigin(images[0].GetOrigin())
			softmax_tfm[channel].SetDirection(images[0].GetDirection())
			softmax_tfm[channel].SetSpacing(images[0].GetSpacing())

		# resample the label back to original space
		resampler = sitk.ResampleImageFilter()
		resampler.SetInterpolator(sitk.sitkNearestNeighbor)
		resampler.SetOutputSpacing(input_spacing)
		resampler.SetSize(input_size)
		resampler.SetOutputOrigin(input_origin)
		resampler.SetOutputDirection(input_direction)

		print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
		label = resampler.Execute(label_tfm)

		if not self.evaluate_probability_output:
			return label

		if self.evaluate_probability_output:
			resampler.SetInterpolator(sitk.sitkLinear)
			print("{}: Resampling probability map back to original image space...".format(datetime.datetime.now()))
			for channel in range(self.output_channel_num):
				softmax_tfm[channel] = resampler.Execute(softmax_tfm[channel])

		return label, softmax_tfm

	def evaluate_single_2D(self,sample, transforms):
		input_origin = sample['image'][0].GetOrigin()
		input_direction = sample['image'][0].GetDirection()
		input_spacing = sample['image'][0].GetSpacing()
		input_size = sample['image'][0].GetSize()

		for transform in transforms['3D']:
			sample = transform(sample)

		images = sample['image']
		label = sample['label']

		if self.evaluate_probability_output:
			prob = []
			for channel in range(self.output_channel_num):
				# create empty softmax image in pair with transformed image
				prob_ = sitk.Image(images[0].GetSize(),sitk.sitkFloat32)
				prob_.SetOrigin(images[0].GetOrigin())
				prob_.SetDirection(images[0].GetDirection())
				prob_.SetSpacing(images[0].GetSpacing())
				prob.append(prob_)

		# loop over slices
		for layer in tqdm(range(images[0].GetSize()[2])):
			# print(str(layer),"/",str(images[0].GetSize()[2]))
			extractor = sitk.ExtractImageFilter()
			size = [images[0].GetSize()[0],images[0].GetSize()[1],0]
			index = [0,0,layer]
			extractor.SetSize(size)
			extractor.SetIndex(index)

			# extract a slice and convert to numpy array
			images_slice = []
			for channel in range(len(images)):
				images_slice.append(extractor.Execute(images[channel]))
			label_slice = extractor.Execute(label)

			sample = {'image':images_slice, 'label':label_slice}

			input_slice_spacing = label_slice.GetSpacing()
			input_slice_direction = label_slice.GetDirection()
			input_slice_size = label_slice.GetSize()
			input_slice_origin = label_slice.GetOrigin()

			for transform in transforms['2D']:
				sample = transform(sample)

			images_slice = sample['image']
			label_slice = sample['label']

			for channel in range(len(images_slice)):
				image_ = sitk.GetArrayFromImage(images_slice[channel])
				if channel == 0:
					images_np = image_[:,:,np.newaxis]
				else:
					images_np = np.append(images_np, image_[:,:,np.newaxis],axis=-1)

			images_np = np.asarray(images_np, np.float32)
			label_np = sitk.GetArrayFromImage(label_slice)
			label_np = np.asarray(label_np, np.int32)

			prob_np = []
			for channel in range(self.output_channel_num):
				prob_np_ = np.zeros(label_np.shape)
				prob_np.append(np.asarray(prob_np_,np.float32))

			# a weighting matrix will be used for averaging the overlapped region
			weight_np = np.zeros(label_np.shape)

			# prepare image batch indices
			inum = int(math.ceil((images_np.shape[0]-self.patch_shape[0])/float(self.evaluate_stride[0]))) + 1 
			jnum = int(math.ceil((images_np.shape[1]-self.patch_shape[1])/float(self.evaluate_stride[1]))) + 1

			patch_total = 0
			image_ij_patch_indices_dicts = []
			ij_patch_indicies_tmp = []

			for i in range(inum):
				for j in range(jnum):
					if patch_total % self.evaluate_batch == 0:
						ij_patch_indicies_tmp = []

					istart = i * self.evaluate_stride[0]
					if istart + self.patch_shape[0] > images_np.shape[0]: #for last patch
						istart = images_np.shape[0] - self.patch_shape[0]
					iend = istart + self.patch_shape[0]

					jstart = j * self.evaluate_stride[1]
					if jstart + self.patch_shape[1] > images_np.shape[1]: #for last patch
						jstart = images_np.shape[1] - self.patch_shape[1]
					jend = jstart + self.patch_shape[1]

					image_patch = images_np[istart:iend,jstart:jend,:]
					image_batch = image_patch[np.newaxis,:,:,:]

					[pred, softmax] = self.sess.run(['predicted_label/prediction:0','softmax:0'], feed_dict={
						'images_placeholder:0': image_batch, 
						'dropout_placeholder:0': 0.0,
						'train_phase_placeholder:0': True})

					for channel in range(self.output_channel_num):
						prob_np[channel][istart:iend,jstart:jend] += softmax[0,:,:,channel]
					weight_np[istart:iend,jstart:jend] += 1.0

			# eliminate overlapping region using the weighted value
			label_np = np.argmax(prob_np,axis=0)

			if self.evaluate_probability_output:
				for channel in range(self.output_channel_num):
					prob_np[channel] = prob_np[channel]/np.float32(weight_np)

			# convert label numpy back to sitk image
			label_slice = sitk.GetImageFromArray(label_np)
			label_slice.SetOrigin(images_slice[0].GetOrigin())
			label_slice.SetDirection(images_slice[0].GetDirection())
			label_slice.SetSpacing(images_slice[0].GetSpacing())

			# resample the label back to original space
			resampler = sitk.ResampleImageFilter()
			resampler.SetInterpolator(sitk.sitkNearestNeighbor)
			resampler.SetOutputSpacing(input_slice_spacing)
			resampler.SetSize(input_slice_size)
			resampler.SetOutputOrigin(input_slice_origin)
			resampler.SetOutputDirection(input_slice_direction)
			label_slice = resampler.Execute(label_slice)

			label_slice = sitk.JoinSeries(label_slice)
			castFilter = sitk.CastImageFilter()
			castFilter.SetOutputPixelType(sitk.sitkUInt8)
			label_slice = castFilter.Execute(label_slice)

			label = sitk.Paste(label,label_slice, label_slice.GetSize(), destinationIndex=[0,0,layer])

			if self.evaluate_probability_output:
				for channel in range(self.output_channel_num):
					prob_slice = sitk.GetImageFromArray(prob_np[channel])
					prob_slice.SetOrigin(images_slice[0].GetOrigin())
					prob_slice.SetDirection(images_slice[0].GetDirection())
					prob_slice.SetSpacing(images_slice[0].GetSpacing())

					# resample the label back to original space
					resampler.SetInterpolator(sitk.sitkLinear)
					prob_slice = resampler.Execute(prob_slice)

					prob_slice = sitk.JoinSeries(prob_slice)
					prob[channel] = sitk.Paste(prob[channel], prob_slice, prob_slice.GetSize(), destinationIndex=[0,0,layer])

		if not self.evaluate_probability_output:
			return label
		else:
			return label, prob

	def evaluate(self):
		# read config to class variables
		self.read_config()

		"""evaluate the vnet model by stepwise moving along the 3D image"""
		# restore model graph and checkpoint
		# tf.reset_default_graph()
		imported_meta = tf.train.import_meta_graph(self.checkpoint_path + ".meta")
		imported_meta.restore(self.sess, self.checkpoint_path)
		print("{}: Restore checkpoint success".format(datetime.datetime.now()))

		# load the pipeline from yaml
		with open(self.evaluate_pipeline) as f:
			pipeline_ = yaml.load(f)

		if self.dimension == 2:
			transforms3D = []
			transforms2D = []

			if pipeline_["preprocess"]["evaluate"]["3D"] is not None:
				for transform in pipeline_["preprocess"]["evaluate"]["3D"]:
					tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
					transforms3D.append(tfm_cls)

			if pipeline_["preprocess"]["evaluate"]["2D"] is not None:
				for transform in pipeline_["preprocess"]["evaluate"]["2D"]:
					tfm_cls = getattr(NiftiDataset2D,transform["name"])(*[],**transform["variables"])
					transforms2D.append(tfm_cls)

			transforms = {'3D': transforms3D, '2D': transforms2D}
		else:
			# create transformation on images and labels
			transforms = []
			if pipeline_["preprocess"]["evaluate"]["3D"] is not None:
				for transform in pipeline_["preprocess"]["evaluate"]["3D"]:
					tfm_cls = getattr(NiftiDataset3D,transform["name"])(*[],**transform["variables"])
					transforms.append(tfm_cls)

		# start evaluation
		print("{}: Start evaluation...".format(datetime.datetime.now()))

		for case in os.listdir(self.evaluate_data_dir):
			# check image data exists
			image_paths = []
			image_file_exists = True
			for channel in range(self.input_channel_num):
				image_paths.append(os.path.join(self.evaluate_data_dir,case,self.evaluate_image_filenames[channel]))
				if not os.path.exists(image_paths[channel]):
					image_file_exists = False
					break
			if not image_file_exists:
				print("{}: Image file not found at {}".format(datetime.datetime.now(),os.path.dirname(image_paths[0])))
				break

			print("{}: Evaluating image at {}".format(datetime.datetime.now(),os.path.dirname(image_paths[0])))

			# read image file
			images = []
			images_tfm = []

			reader = sitk.ImageFileReader()
			for channel in range(self.input_channel_num):
				reader.SetFileName(image_paths[channel])
				image = reader.Execute()
				images.append(image)
				images_tfm.append(image)

			# create empty label in pair with the transformed image
			label_tfm = sitk.Image(images[0].GetSize(),sitk.sitkUInt8)
			label_tfm.SetOrigin(images[0].GetOrigin())
			label_tfm.SetDirection(images[0].GetDirection())
			label_tfm.SetSpacing(images[0].GetSpacing())

			sample = {'image':images_tfm, 'label': label_tfm}

			if self.evaluate_probability_output:
				if self.dimension == 2:
					label, prob = self.evaluate_single_2D(sample,transforms)
				else:
					label, prob = self.evaluate_single_3D(sample,transforms)
			else:
				if self.dimension == 2:
					label = self.evaluate_single_2D(sample,transforms)
				else:
					label = self.evaluate_single_3D(sample,transforms)

			# largest connected component
			if self.evaluate_lcc:
				label = ExtractLargestConnectedComponents(label)

			# volume threshold
			if self.evaluate_volume_threshold > 0:
				label = volume_threshold(label,self.evaluate_volume_threshold)

			# save segmented label
			writer = sitk.ImageFileWriter()
			
			label_path = os.path.join(self.evaluate_data_dir,case,self.evaluate_label_filename)
			writer.SetFileName(label_path)
			writer.Execute(label)

			tqdm.write("{}: Save evaluate label at {} success".format(datetime.datetime.now(),label_path))

			if self.evaluate_probability_output:
				for channel in range(self.output_channel_num):
					ext = ""
					for ext_ in self.evaluate_probability_filename.split(".")[1:]:
						ext += "." + ext_
					output_filename = self.evaluate_probability_filename.split(".")[0] + "_" + str(self.label_classes[channel]) + ext
					prob_path = os.path.join(self.evaluate_data_dir,case,output_filename)
					writer.SetFileName(prob_path)
					writer.Execute(prob[channel])
					tqdm.write("{}: Save evaluate probability map at {} success".format(datetime.datetime.now(),prob_path))
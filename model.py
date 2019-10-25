import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import NiftiDataset3D
import sys
import datetime
import numpy as np

def grayscale_to_rainbow(image):
	# grayscale to rainbow colormap, convert to HSV (H = reversed grayscale from 0:2/3, S and V are all 1)
	# then convert to RGB
	H = tf.squeeze((1. - image)*2./3., axis=-1)
	SV = tf.ones(H.get_shape())
	HSV = tf.stack([H,SV,SV], axis=3)
	RGB = tf.image.hsv_to_rgb(HSV)

	return RGB

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

		self.restore_training = self.config['TrainingSetting']['Restore']
		self.log_dir = self.config['TrainingSetting']['LogDir']
		self.ckpt_dir = self.config['TrainingSetting']['CheckpointDir']
		self.epoches = self.config['TrainingSetting']['Epoches']

		self.network_name = self.config['TrainingSetting']['Networks']['Name']
		self.attention = self.config['TrainingSetting']['Networks']['Attention']
		self.dropout_rate = self.config['TrainingSetting']['Networks']['Dropout']

		self.optimizer_name = self.config['TrainingSetting']['Optimizer']['Name']
		self.initial_learning_rate = self.config['TrainingSetting']['Optimizer']['InitialLearningRate']
		self.decay_factor = self.config['TrainingSetting']['Optimizer']['Decay']['Factor']
		self.decay_step = self.config['TrainingSetting']['Optimizer']['Decay']['Step']
		self.spacing = self.config['TrainingSetting']['Spacing']
		self.drop_ratio = self.config['TrainingSetting']['DropRatio']
		self.min_pixel = self.config['TrainingSetting']['MinPixel']

	def placeholder_inputs(self, input_batch_shape, output_batch_shape, attention=False):
		"""Generate placeholder variables to represent the the input tensors.
		These placeholders are used as inputs by the rest of the model building
		code and will be fed from the downloaded ckpt in the .run() loop, below.
		Args:
			patch_shape: The patch_shape will be baked into both placeholders.
			attention: Boolean to determine if the model consist of attention module (default=False)
		Returns:
			images_placeholder: Images placeholder.
			labels_placeholder: Labels placeholder.
			distmap_placeholder: Attention map placeholder.
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

	def dataset_iterator(self, data_dir, transforms, train=True):
		# Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
		with tf.device('/cpu:0'):
			if self.dimension==2:
				Dataset = NiftiDataset2D.NiftiDataset()
			else:
				Dataset = NiftiDataset3D.NiftiDataset(
					data_dir=data_dir,
					image_filenames=self.image_filenames,
					label_filename=self.label_filename,
					transforms=transforms,
					train=True,
					distmap=self.attention
				)
			
			dataset = Dataset.get_dataset()
			dataset = dataset.shuffle(buffer_size=5)
			dataset = dataset.batch(self.batch_size,drop_remainder=True)

		return dataset.make_initializable_iterator()

	def build_model_graph(self):
		self.global_step_op = tf.train.get_or_create_global_step()		

		if self.dimension == 2:
			input_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.input_channel_num) 
			output_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], 1) 
		elif self.dimension == 3:
			input_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], self.input_channel_num) 
			output_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], 1) 
		else:
			sys.exit('Invalid Patch Shape (length should be 2 or 3)')

		if self.attention:
			self.images_placeholder, self.labels_placeholder, self.distmap_placeholder = self.placeholder_inputs(input_batch_shape,output_batch_shape, attention=True)
		else:
			self.images_placeholder, self.labels_placeholder = self.placeholder_inputs(input_batch_shape,output_batch_shape, attention=False)

		# plot input and output images to tensorboard
		if self.image_log:
			if self.dimension == 2:
				for image_channel in range(self.input_channel_num):
					image_log = tf.cast(self.images_placeholder[:,:,:,image_channel:image_channel+1], dtype=tf.uint8)
					tf.summary.image(self.image_filenames[image_channel], image_log, max_outputs=self.batch_size)
				if self.attention:
					distmap_log = grayscale_to_rainbow(distmap_placeholder,dtype=tf.uint8)
					tf.summary.image("distmap", distmap_log, max_outputs=self.batch_size)
				labels_log = tf.cast(self.labels_placeholder*255, dtype=tf.uint8)
				tf.summary.image("label",labels_log, max_outputs=self.batch_size)
			else:
				for batch in range(self.batch_size):
					for image_channel in range(self.input_channel_num):
						image_log = tf.cast(self.images_placeholder[batch:batch+1,:,:,:,image_channel], dtype=tf.uint8)
						tf.summary.image(self.image_filenames[image_channel], tf.transpose(image_log,[3,1,2,0]),max_outputs=self.patch_shape[-1])
					if self.attention:
						distmap_log = grayscale_to_rainbow(tf.transpose(distmap_placeholder[batch:batch+1,:,:,:,0],[3,1,2,0]))
						tf.summary.image("distmap", distmap_log, max_outputs=self.patch_shape[-1])
					labels_log = tf.cast(self.labels_placeholder[batch:batch+1,:,:,:,0]*255,dtype=tf.uint8)
					tf.summary.image("label", tf.transpose(labels_log,[3,1,2,0]),max_outputs=self.patch_shape[-1])

		# Get images and labels
		# support multiple image input, but here only use single channel, label file should be a single file with different classes

		# create transformations to image and labels
		if self.dimension == 2:
			sys.exit("2D training under development")
		else:
			trainTransforms = [
				NiftiDataset3D.ExtremumNormalization(0.1),
				# NiftiDataset.Normalization(),
				NiftiDataset3D.Resample((0.25,0.25,0.25)),
				NiftiDataset3D.Padding((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2])),
				NiftiDataset3D.RandomCrop((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),self.drop_ratio, self.min_pixel),
				# NiftiDataset.ConfidenceCrop((FLAGS.patch_size*3, FLAGS.patch_size*3, FLAGS.patch_layer*3),(0.0001,0.0001,0.0001)),
				# NiftiDataset.BSplineDeformation(randomness=2),
				# NiftiDataset.ConfidenceCrop((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),(0.5,0.5,0.5)),
				# NiftiDataset3D.ConfidenceCrop2((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),rand_range=32,probability=0.8),
				# NiftiDataset3D.RandomFlip([True, False, False]),
				NiftiDataset3D.RandomNoise()
				]

			# use random crop for testing
			testTransforms = [
				NiftiDataset3D.ExtremumNormalization(0.1),
				# NiftiDataset.Normalization(),
				NiftiDataset3D.Resample((0.25,0.25,0.25)),
				NiftiDataset3D.Padding((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2])),
				NiftiDataset3D.RandomCrop((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),self.drop_ratio, self.min_pixel)
				# NiftiDataset.ConfidenceCrop((FLAGS.patch_size*2, FLAGS.patch_size*2, FLAGS.patch_layer*2),(0.0001,0.0001,0.0001)),
				# NiftiDataset.BSplineDeformation(),
				# NiftiDataset.ConfidenceCrop((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),(0.75,0.75,0.75)),
				# NiftiDataset.ConfidenceCrop2((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),rand_range=32,probability=0.8),
				# NiftiDataset.RandomFlip([True, False, False]),
				]

		# get input and output datasets
		self.train_iterator = self.dataset_iterator(self.train_data_dir, trainTransforms)
		self.next_element_train = self.train_iterator.get_next()

		if self.testing:
			self.test_iterator = self.dataset_iterator(self.test_data_dir, testTransforms)
			self.next_element_test = self.test_iterator.get_next()

	def train(self):
		# read config to class variables
		self.read_config()

		"""Train image2label model"""
		self.build_model_graph()

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

		# loop over epochs
		for epoch in np.arange(start_epoch.eval(session=self.sess), self.epoches):
			print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch+1))
			# initialize iterator in each new epoch
			self.sess.run(self.train_iterator.initializer)
			if self.testing:
				self.sess.run(self.test_iterator.initializer)

			# training phase
			while True:
				try:
					self.sess.run(tf.local_variables_initializer())
					if self.attention:
						image, label, distmap = self.sess.run(self.next_element_train)
						distmap = distmap[:,:,:,:,np.newaxis]
					else:
						image, label = self.sess.run(self.next_element_train)

					label = label[:,:,:,:,np.newaxis]

					if self.attention:
						summary = self.sess.run(summary_op, feed_dict={
							self.images_placeholder: image,
							self.labels_placeholder: label,
							self.distmap_placeholder: distmap
							})
					else:
						summary = self.sess.run(summary_op, feed_dict={
							self.images_placeholder: image,
							self.labels_placeholder: label
							})

					train_summary_writer.add_summary(summary,global_step=tf.train.global_step(self.sess,self.global_step_op))
					train_summary_writer.flush()
				except tf.errors.OutOfRangeError:
					break

	def evaluate(self):
		sys.exit("evaluation under development")
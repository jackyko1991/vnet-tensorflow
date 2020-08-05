from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import NiftiDataset3D as NiftiDataset
import os
import datetime
import SimpleITK as sitk
import math
import numpy as np
from tqdm import tqdm
import json
import multiprocessing

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','./data_WML/evaluate',
	"""Directory of evaluation data""")
tf.app.flags.DEFINE_string('config_json','./config.json',
	"""JSON file for filename configuration""")
tf.app.flags.DEFINE_string('model_path','./tmp/ckpt/checkpoint-103233.meta',
	"""Path to saved models""")
tf.app.flags.DEFINE_string('checkpoint_path','./tmp/ckpt/checkpoint-103233',
	"""Directory of saved checkpoints""")
tf.app.flags.DEFINE_integer('patch_size',192,
	"""Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',16,
	"""Number of layers in data patch""")
tf.app.flags.DEFINE_integer('stride_inplane', 144,
	"""Stride size in 2D plane""")
tf.app.flags.DEFINE_integer('stride_layer',12,
	"""Stride size in layer direction""")
tf.app.flags.DEFINE_integer('batch_size',5,
	"""Setting batch size (currently only accept 1)""")
tf.app.flags.DEFINE_boolean('attention',False,
	"""Set if the training model applies attention module""")

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

def evaluate():
	"""evaluate the vnet model by stepwise moving along the 3D image"""
	# restore model grpah
	tf.reset_default_graph()
	imported_meta = tf.train.import_meta_graph(FLAGS.model_path)

	# read configuration file
	with open(FLAGS.config_json) as config_json:  
		json_config = json.load(config_json)

	input_channel_num = len(json_config['TrainingSetting']['Data']['ImageFilenames'])

	# create transformations to image and labels
	transforms = [  
		NiftiDataset.StatisticalNormalization(3.0, pre_norm=True),
		# NiftiDataset.Normalization(),
		NiftiDataset.Resample((json_config['TrainingSetting']['Spacing'][0],json_config['TrainingSetting']['Spacing'][1],json_config['TrainingSetting']['Spacing'][2])),
		NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
		]

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:  
		print("{}: Start evaluation...".format(datetime.datetime.now()))

		imported_meta.restore(sess, FLAGS.checkpoint_path)
		print("{}: Restore checkpoint success".format(datetime.datetime.now()))
		
		for case in os.listdir(FLAGS.data_dir):
			# ops to load data
			# support multiple image input, but here only use single channel, label file should be a single file with different classes

			# check image data exists
			image_paths = []
			image_file_exists = True
			for image_channel in range(input_channel_num):
				image_paths.append(os.path.join(FLAGS.data_dir,case,json_config['EvaluationSetting']['Data']['ImageFilenames'][image_channel]))

				if not os.path.exists(image_paths[image_channel]):
					image_file_exists = False
					break

			if not image_file_exists:
				# print("{}: Image file not found at {}".format(datetime.datetime.now(),os.path.dirname(image_paths[0])))
				print("{}: Image file not found at {}".format(datetime.datetime.now(),image_paths[0]))
				break

			print("{}: Evaluating image at {}".format(datetime.datetime.now(),os.path.dirname(image_paths[0])))

			# read image file
			images = []
			image_tfm = []
			
			for image_channel in range(input_channel_num):
				reader = sitk.ImageFileReader()
				reader.SetFileName(image_paths[image_channel])
				image = reader.Execute()
				images.append(image)
				# preprocess the image and label before inference
				image_tfm.append(image)

			# create empty label in pair with transformed image
			label_tfm = sitk.Image(image_tfm[0].GetSize(),sitk.sitkUInt32)
			label_tfm.SetOrigin(image_tfm[0].GetOrigin())
			label_tfm.SetDirection(image_tfm[0].GetDirection())
			label_tfm.SetSpacing(image_tfm[0].GetSpacing())

			sample = {'image':image_tfm, 'label': label_tfm}

			for transform in transforms:
				sample = transform(sample)

			image_tfm, label_tfm = sample['image'], sample['label']

			# create empty softmax image in pair with transformed image
			softmax_tfm = sitk.Image(image_tfm[0].GetSize(),sitk.sitkFloat32)
			softmax_tfm.SetOrigin(image_tfm[0].GetOrigin())
			softmax_tfm.SetDirection(image_tfm[0].GetDirection())
			softmax_tfm.SetSpacing(image_tfm[0].GetSpacing())

			# convert image to numpy array
			for image_channel in range(input_channel_num):
				image_ = sitk.GetArrayFromImage(image_tfm[image_channel])
				if image_channel == 0:
					images_np = image_[:,:,:,np.newaxis]
				else:
					images_np = np.append(images_np, image_[:,:,:,np.newaxis], axis=-1)

			images_np = np.asarray(images_np,np.float32)

			label_np = sitk.GetArrayFromImage(label_tfm)
			label_np = np.asarray(label_np,np.int32)

			softmax_np = sitk.GetArrayFromImage(softmax_tfm)
			softmax_np = np.asarray(softmax_np,np.float32)

			# unify numpy and sitk orientation
			images_np = np.transpose(images_np,(2,1,0,3))
			label_np = np.transpose(label_np,(2,1,0))
			softmax_np = np.transpose(softmax_np,(2,1,0))

			# a weighting matrix will be used for averaging the overlapped region
			weight_np = np.zeros(label_np.shape)

			# prepare image batch indices
			inum = int(math.ceil((images_np.shape[0]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1 
			jnum = int(math.ceil((images_np.shape[1]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1
			knum = int(math.ceil((images_np.shape[2]-FLAGS.patch_layer)/float(FLAGS.stride_layer))) + 1

			patch_total = 0
			image_ijk_patch_indices_dicts = []
			ijk_patch_indicies_tmp = []

			for i in range(inum):
				for j in range(jnum):
					for k in range (knum):
						if patch_total % FLAGS.batch_size == 0:
							ijk_patch_indicies_tmp = []

						istart = i * FLAGS.stride_inplane
						if istart + FLAGS.patch_size > images_np.shape[0]: #for last patch
							istart = images_np.shape[0] - FLAGS.patch_size 
						iend = istart + FLAGS.patch_size

						jstart = j * FLAGS.stride_inplane
						if jstart + FLAGS.patch_size > images_np.shape[1]: #for last patch
							jstart = images_np.shape[1] - FLAGS.patch_size 
						jend = jstart + FLAGS.patch_size

						kstart = k * FLAGS.stride_layer
						if kstart + FLAGS.patch_layer > images_np.shape[2]: #for last patch
							kstart = images_np.shape[2] - FLAGS.patch_layer 
						kend = kstart + FLAGS.patch_layer

						ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

						if patch_total % FLAGS.batch_size == 0:
							image_ijk_patch_indices_dicts.append({'images': images_np, 'indexes':ijk_patch_indicies_tmp})

						patch_total += 1
			
			# for last batch
			image_ijk_patch_indices_dicts.append({'images': images_np, 'indexes':ijk_patch_indicies_tmp})

			p = multiprocessing.Pool(multiprocessing.cpu_count())
			batches = p.map(prepare_batch,image_ijk_patch_indices_dicts)
			p.close()
			p.join()

			# batches = prepare_batch(images_np,ijk_patch_indices)

			# acutal segmentation
			for i in tqdm(range(len(batches))):
				batch = batches[i]
				if FLAGS.attention:
					[pred, softmax] = sess.run(['predicted_label/prediction:0','softmax:0'], feed_dict={
						'images_placeholder:0': batch, 
						'train_phase_placeholder:0': False,
						'attention/train_phase_placeholder:0': False,
						'output/train_phase_placeholder:0': False})
				else:
					[pred, softmax] = sess.run(['predicted_label/prediction:0','softmax:0'], feed_dict={
						'images_placeholder:0': batch, 
						'train_phase_placeholder:0': False})

				for j in range(pred.shape[0]):
					istart = image_ijk_patch_indices_dicts[i]['indexes'][j][0]
					iend = image_ijk_patch_indices_dicts[i]['indexes'][j][1]
					jstart = image_ijk_patch_indices_dicts[i]['indexes'][j][2]
					jend = image_ijk_patch_indices_dicts[i]['indexes'][j][3]
					kstart = image_ijk_patch_indices_dicts[i]['indexes'][j][4]
					kend = image_ijk_patch_indices_dicts[i]['indexes'][j][5]
					label_np[istart:iend,jstart:jend,kstart:kend] += pred[j,:,:,:]
					softmax_np[istart:iend,jstart:jend,kstart:kend] += softmax[j,:,:,:,1]
					weight_np[istart:iend,jstart:jend,kstart:kend] += 1.0

			print("{}: Evaluation complete".format(datetime.datetime.now()))
			# eliminate overlapping region using the weighted value
			label_np = np.rint(np.float32(label_np)/np.float32(weight_np) + 0.01)
			softmax_np = softmax_np/np.float32(weight_np)

			# convert back to sitk space
			label_np = np.transpose(label_np,(2,1,0))
			softmax_np = np.transpose(softmax_np,(2,1,0))

			# convert label numpy back to sitk image
			label_tfm = sitk.GetImageFromArray(label_np)
			label_tfm.SetOrigin(image_tfm[0].GetOrigin())
			label_tfm.SetDirection(image_tfm[0].GetDirection())
			label_tfm.SetSpacing(image_tfm[0].GetSpacing())

			softmax_tfm = sitk.GetImageFromArray(softmax_np)
			softmax_tfm.SetOrigin(image_tfm[0].GetOrigin())
			softmax_tfm.SetDirection(image_tfm[0].GetDirection())
			softmax_tfm.SetSpacing(image_tfm[0].GetSpacing())

			# resample the label back to original space
			resampler = sitk.ResampleImageFilter()
			# save segmented label
			writer = sitk.ImageFileWriter()

			resampler.SetInterpolator(1)
			resampler.SetOutputSpacing(images[0].GetSpacing())
			resampler.SetSize(images[0].GetSize())
			resampler.SetOutputOrigin(images[0].GetOrigin())
			resampler.SetOutputDirection(images[0].GetDirection())
			
			print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
			label = resampler.Execute(label_tfm)
			castFilter = sitk.CastImageFilter()
			castFilter.SetOutputPixelType(sitk.sitkUInt16)
			label = castFilter.Execute(label)

			# volume thresholding
			if json_config['EvaluationSetting']['VolumeThreshold'] > 0:
				label = volume_threshold(label,json_config['EvaluationSetting']['VolumeThreshold'])

			label_path = os.path.join(FLAGS.data_dir,case,json_config['EvaluationSetting']['Data']['LabelFilename'])
			writer.SetFileName(label_path)
			writer.Execute(label)

			print("{}: Save evaluate label at {} success".format(datetime.datetime.now(),label_path))

			print("{}: Resampling probability map back to original image space...".format(datetime.datetime.now()))
			prob = resampler.Execute(softmax_tfm)

			if json_config['EvaluationSetting']['VolumeThreshold'] > 0:
				label = volume_threshold(label,json_config['EvaluationSetting']['VolumeThreshold'])
				castFilter.SetOutputPixelType(sitk.sitkFloat32)
				label = castFilter.Execute(label)
				maskFilter = sitk.MaskNegatedImageFilter()
				prob = maskFilter.Execute(prob,label)

			prob_path = os.path.join(FLAGS.data_dir,case,json_config['EvaluationSetting']['Data']['ProbabilityFilename'])
			writer.SetFileName(prob_path)
			writer.Execute(prob)
			print("{}: Save evaluate probability map at {} success".format(datetime.datetime.now(),prob_path))

def main(argv=None):
	evaluate()

if __name__=='__main__':
	tf.app.run()
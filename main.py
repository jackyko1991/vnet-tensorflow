import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from model import image2label
import tensorflow as tf
import shutil
import json

def str2bool(v):
	#susendberg's function
	return v.lower() in ("yes", "true", "t", "1")

def get_parser():
	# create parser object
	parser = argparse.ArgumentParser(description='Tensorflow implementation for segmentation, specialized for medical image purpose.',
		epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

	# register type keyword to registries
	parser.register('type','bool',str2bool)

	# add arguments
	parser.add_argument(
		'-v', '--verbose',
		dest='verbose',
		help='Show verbose output',
		action='store_true')
	parser.add_argument(
		'-p','--phase', 
		dest='phase', 
		help='Training phase (default= train)',
		choices=['train','evaluate'],
		default='train',
		metavar='[train evaluate]')
	parser.add_argument(
		'--config_json',
		dest='config_json',
		help='JSON file for model configuration',
		type=str,
		default='config.json', 
		metavar='FILENAME'
		)
	parser.add_argument(
		'--gpu',
		dest='gpu',
		default='0',
		type=str,
		help='Select GPU device(s) (default = 0)',
		metavar='GPU_IDs')

	args = parser.parse_args()

	# print arguments if verbose
	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	# select gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # e.g. "0,1,2", "0,2" 

	# read configuration file
	with open(args.config_json) as config_json:
		config = json.load(config_json)

	# session config
	config_proto = tf.ConfigProto()
	config_proto.gpu_options.allow_growth = True

	with tf.Session(config=config_proto) as sess:
		model = image2label(sess,config)
		if args.phase == "train":
			model.train()
		elif args.phase == "evaluate":
			model.evaluate()
		else:
			sys.exit("Invalid training phase")

if __name__=="__main__":
	args = get_parser()
	main(args)
import batch_evaluate
import argparse
import os

def readable_dir(directory):
	"""
	'Type' for argparse - checks that directory exists.
	"""
	if not os.path.isdir(directory):
		raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(directory))
	if not os.access(directory, os.R_OK):
		raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(directory))

	return directory

def get_args():
	parser = argparse.ArgumentParser(description="Batch evaluation tool for VNet.",
		epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

	parser.add_argument(
		'-v', '--verbose',    
		dest='verbose',  
		help='Show verbose output', 
		action='store_true')  
	parser.add_argument(
		'-cd', '--checkpoint_dir',
		dest='checkpoint_dir',
		help='Checkpoint folder',
		type=readable_dir,
		metavar='DIR',
		default='./tmp/ckpt',
		required=True
		)
	parser.add_argument(
		'-cmin','--checkpoint_min',
		dest='checkpoint_min',
		help='Minimum checkpoint number',
		type=int,
		metavar='INT',
		default=10000
		)
	parser.add_argument(
		'-cmax','--checkpoint_max',
		dest='checkpoint_max',
		help='Maximum checkpoint number',
		type=int,
		metavar='INT',
		default=999999999999999999999999999999)
	parser.add_argument(
		'-slmin','--stride_layer_min',
		dest='stride_layer_min',
		help='Minimum stride across layers',
		type=int,
		metavar='INT',
		default=32)
	parser.add_argument(
		'-slmax','--stride_layer_max',
		dest='stride_layer_max',
		help='Maximum stride across layers',
		type=int,
		metavar='INT',
		default=64)
	parser.add_argument(
		'-spmin','--stride_inplane_min',
		dest='stride_inplane_min',
		help='Minimum stride across the plane',
		type=int,
		metavar='INT',
		default=32)
	parser.add_argument(
		'-spmax','--stride_inplane_max',
		dest='stride_inplane_max',
		help='Maximum stride across the plane',
		type=int,
		metavar='INT',
		default=64)
	parser.add_argument(
		'-ps','--patch_size',
		dest='patch_size',
		help='Patch size',
		type=int,
		metavar='INT',
		default=64)
	parser.add_argument(
		'-pl','--patch_layer',
		dest='patch_layer',
		help='Patch layer',
		type=int,
		metavar='INT',
		default=64)
	parser.add_argument(
		'-b','--batch',
		dest='batch',
		help='Batch size',
		type=int,
		metavar='INT',
		default=5)
	parser.add_argument(
		'-d','--data_dir',
		dest='data_dir',
		help='Data folder',
		type=readable_dir,
		metavar='DIR',
		default='./data/evaluate')
	parser.add_argument(
		'-gt', '--ground_truth_filename',
		dest='ground_truth_filename',
		help='Ground truth filename',
		type=str,
		metavar='FILENAME',
		default="./3DRA_seg.nii.gz"
		)
	parser.add_argument(
		'-e', '--evaluate_filename',
		dest='evaluate_filename',
		help='Evaluated filename',
		type=str,
		metavar='FILENAME',
		default="./label_tf.nii.gz"
		)
	parser.add_argument(
		'-c', '--config_json',
		dest='config_json',
		help='Config JSON file',
		type=str,
		metavar='FILENAME',
		default="./config.json"
		)
	parser.add_argument(
		'-o', '--output',
		dest='output',
		help='Output directory',
		type=readable_dir,
		metavar='DIR',
		default="./tmp"
		)

	params = '--checkpoint_min 30000 \
		--stride_inplane_min 118 \
		--stride_inplane_max 128 \
		--stride_layer_min 118 \
		--stride_layer_max 128 \
		--patch_size 128 \
		--patch_layer 128 \
		--batch 1 \
		--data_dir /home/jacky/Projects/vnet-tensorflow/data_3DRA/evaluate \
		--checkpoint_dir /home/jacky/Projects/vnet-tensorflow/tmp_3DRA_0.25/ckpt \
		--output /home/jacky/Projects/vnet-tensorflow/tmp_3DRA_0.25 \
		--config_json  /home/jacky/Projects/vnet-tensorflow/config_3DRA.json \
	'
	args = parser.parse_args(params.split())
	# args = parser.parse_args()

	# print arguments if verbose
	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	be = batch_evaluate.Batch_Evaluate()
	be.config_json = args.config_json
	be.model_folder = args.checkpoint_dir
	be.checkpoint_min = args.checkpoint_min
	be.checkpoint_max = args.checkpoint_max
	be.stride_layer_min = args.stride_layer_min
	be.stride_layer_max = args.stride_layer_max
	be.stride_inplane_min = args.stride_inplane_min
	be.stride_inplane_max = args.stride_inplane_max
	be.patch_size = args.patch_size
	be.patch_layer = args.patch_layer
	be.batch_size = args.batch
	be.data_folder = args.data_dir
	be.ground_truth_filename = args.ground_truth_filename
	be.evaluated_filename = args.evaluate_filename
	be.output_folder = args.output
	be.Execute()

	return

if __name__=="__main__":
	args = get_args()
	main(args)
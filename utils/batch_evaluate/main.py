import batch_evaluate
import argparse

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

	params = '--checkpoint_min 20000 \
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
	be.model_folder = args.checkpoint_dir
	be.checkpoint_min = args.checkpoint_min
	be.checkpoint_max = args.checkpoint_max
	be.stride_min = 40
	be.stride_max = 64
	be.batch_size = 10
	be.data_folder = "./data_WML/evaluate"
	be.ground_truth_filename = "./label.nii.gz"
	be.evaluated_filename = "./label_vnet.nii.gz"
	be.output_folder = "./tmp"
	be.Execute()

	return

if __name__=="__main__":
	args = get_args()
	main(args)
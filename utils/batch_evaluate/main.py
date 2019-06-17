import batch_evaluate
import argparse

def get_args():
	parser = argparse.ArgumentParser(description="Batch evaluation tool for VNet.",
		epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

	parser.add_argument(
		'-v', '--verbose',    
		dest='verbose',  
		help='show verbose output', 
		action='store_true')   

	# params = '--registration False \
	# 	--method SVD \
	# 	--dir ./data/selected_cases/case0/nii \
	# 	--spacing 2 2 5 \
	# 	--blur_factor 2 \
	# 	--block_circulant True \
	# 	--pre_filter True'
	# args = parser.parse_args(params.split())
	args = parser.parse_args()

	# print arguments if verbose
	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	be = batch_evaluate.Batch_Evaluate()
	be.model_folder = "./tmp_dental/ckpt"
	be.checkpoint_min = 5000
	be.batch_size = 10
	be.Execute()

	return

if __name__=="__main__":
	args = get_args()
	main(args)
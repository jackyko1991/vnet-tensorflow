import argparse
from bbox import *

def str2bool(v):
	#susendberg's function
	return v.lower() in ("yes", "true", "t", "1")

def get_parser():
	# create parser object
	parser = argparse.ArgumentParser(description='Draw bounding box from NIFTI segmented labels.',
		epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

	# register type keyword to registries
	parser.register('type','bool',str2bool)

	# add arguments
	parser.add_argument(
		'image',
		help='Input NIFTI image',
		type=str,
		metavar='PATH'
		)
	parser.add_argument(
		'label',
		help='Input NIFTI label',
		type=str,
		metavar='PATH'
		)
	parser.add_argument(
		'output',
		help='Output directory',
		default='./output',
		type=str,
		metavar='DIR'
		)
	parser.add_argument(
		'-c',
		dest='classname',
		help='Classname file',
		type=str,
		metavar='TXT'
		)
	parser.add_argument(
		'-f','--format', 
		dest='format', 
		help='Output image format',
		choices=['png','jpg'],
		default='png',
		metavar='[png jpg]'
		)
	parser.add_argument(
		'-v', '--verbose',
		dest='verbose',
		help='Show verbose output',
		action='store_true')
	parser.add_argument(
		'-o','--opacity',
		dest='opacity',
		default=0,
		type=float,
		help='Segmentation label opacity (default=0)',
		metavar='FLOAT [0-1]')
	parser.add_argument(
		'-d','--direction', 
		dest='direction', 
		help='Image series direction (default=axial)',
		choices=['axial','sagittal','coronal'],
		default='axial',
		metavar='[axial sagittal coronal]')
	parser.add_argument(
		'-m','--min',
		dest='min',
		default=-1024,
		type=float,
		help='Minimum image intensity (default=-1024)',
		metavar='FLOAT'
		)
	parser.add_argument(
		'-M','--max',
		dest='max',
		default=1024,
		type=float,
		help='Maximum image intensity (default=1024)',
		metavar='FLOAT'
		)

	args = parser.parse_args()

	# print arguments if verbose
	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	bbox = BoundingBox(args.image, args.label, args.output)
	bbox.min_intensity = args.min
	bbox.max_intensity = args.max
	bbox.opacity = args.opacity
	bbox.classnames = args.classname
	bbox.run()

if __name__=="__main__":
	args = get_parser()
	main(args)
import os
import SimpleITK as sitk
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from tqdm import tqdm

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes   
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,0]+boxes[:,2]
	y2 = boxes[:,1]+boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def bboxes_from_slice(image_slice, label_slice,plot=False, min_intensity=-1024, max_intensity=1024, opacity=0,classnames={},save_path=""):
	labelStatFilter = sitk.LabelStatisticsImageFilter()
	labelStatFilter.Execute(image_slice, label_slice)

	bboxes = []
	for label in labelStatFilter.GetLabels():
		if label == 0:
			continue

		# connected components
		binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
		binaryThresholdFilter.SetLowerThreshold(label)
		binaryThresholdFilter.SetUpperThreshold(label)
		binaryThresholdFilter.SetInsideValue(1)
		binaryThresholdFilter.SetOutsideValue(0)
		label_slice_ = binaryThresholdFilter.Execute(label_slice)

		ccFilter = sitk.ConnectedComponentImageFilter()
		label_slice_ = ccFilter.Execute(label_slice_)

		labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		labelShapeFilter.Execute(label_slice_)

		boxes = []
		for cc_region in labelShapeFilter.GetLabels():
			(x, y, w, h) = labelShapeFilter.GetBoundingBox(cc_region)
			boxes.append([x,y,w,h])
		boxes = np.array(boxes)

		# combine overlapping bboxes
		boxes = non_max_suppression_fast(boxes,0.5)

		for box in boxes:
			bboxes.append((box[0],box[1],box[2],box[3],label))

	# plot to debug
	intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
	intensityWindowingFilter.SetOutputMaximum(255)
	intensityWindowingFilter.SetOutputMinimum(0)
	intensityWindowingFilter.SetWindowMaximum(max_intensity);
	intensityWindowingFilter.SetWindowMinimum(min_intensity);

	image_np = sitk.GetArrayFromImage(intensityWindowingFilter.Execute(image_slice))
	image_np = image_np/255

	# Create figure and axes
	fig,ax = plt.subplots(1)

	# Display the image
	mask = sitk.GetArrayFromImage(label_slice)
	masked = np.ma.masked_where(mask == 0, mask)

	ax.imshow(image_np,cmap="gray")
	ax.imshow(masked,cmap="jet",alpha=opacity)

	ax.set_axis_off()

	# Create a Rectangle patch
	for (x, y, w, h, label) in bboxes:
		if label == 1:
			color = "r"
		elif label == 2:
			color = "c"
		rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=color,facecolor="none")
		if str(label) in classnames.keys():
			ax.text(x,y-3, classnames[str(label)], color="w")

		# Add the patch to the Axes
		ax.add_patch(rect)

	if plot:
		plt.show()

	if save_path != "":
		plt.savefig(save_path,bbox_inches='tight',transparent=True, pad_inches=0)

	plt.clf()
	plt.close()

	return bboxes

class BoundingBox:
	def __init__(self,
		image_path,
		label_path,
		output_dir="./output",
		image_format="png",
		opacity=0,
		direction="axial",
		min_intensity=-1024,
		max_intensity=1024,
		classname_file=""
		):
		self.image_path = image_path
		self.label_path = label_path
		self.output_dir = output_dir
		self.image_format = image_format
		self.opacity = opacity
		self.direction = direction
		self.min_intensity=min_intensity
		self.max_intensity=max_intensity
		self.classname_file=""
		self.classnames={}

	def run(self):
		# check file existence
		if not (os.path.exists(self.image_path) and os.path.exists(self.label_path)):
			raise IOError("Input image/label file not exist")

		# check input corrert
		assert self.image_format in ["png","jpg"], "Output image format can only be png or jpg"
		assert self.opacity>=0 and self.opacity<=1, "Opacity should between 0 and 1"
		assert self.direction in ["axial", "coronal", "sagittal"], "Image direction can only be axial, coronal or sagittal"

		# create output dir
		os.makedirs(self.output_dir,exist_ok=True)

		# read the image and label
		reader = sitk.ImageFileReader()
		reader.SetFileName(self.image_path)
		image = reader.Execute()

		reader.SetFileName(self.label_path)
		label = reader.Execute()

		# get image smallest spacing and convert to isotropic image
		if self.direction == "axial":
			minSpacing = min(image.GetSpacing()[0:2])
		old_spacing = image.GetSpacing()
		old_size = image.GetSize()

		if self.direction == "axial":
			new_spacing = (minSpacing,minSpacing,old_spacing[2])
		new_size = [int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])) for i in range(3)]
		new_size = tuple(new_size)

		resampler = sitk.ResampleImageFilter()
		resampler.SetOutputSpacing(new_spacing)
		resampler.SetSize(new_size)
		resampler.SetOutputOrigin(image.GetOrigin())
		resampler.SetOutputDirection(image.GetDirection())

		resampler.SetInterpolator(sitk.sitkLinear)
		image = resampler.Execute(image)
		resampler.SetInterpolator(sitk.sitkNearestNeighbor)
		label = resampler.Execute(label)

		size = list(image.GetSize())
		size[2] = 0

		# read class name file
		if self.classname_file is not None:
			if self.classname_file != "" and os.path.exists(self.classname_file):

				with open(self.classname_file,"r") as f:
					self.classnames = json.load(f)

		if self.direction == "axial":
			pbar = tqdm(range(image.GetSize()[2])[:])
			for z in pbar:
				index = [0, 0, z]
				extractor = sitk.ExtractImageFilter()
				extractor.SetSize(size)
				extractor.SetIndex(index)

				bboxes = bboxes_from_slice(extractor.Execute(image),extractor.Execute(label),
					plot=False,
					min_intensity=self.min_intensity,
					max_intensity=self.max_intensity,
					opacity=self.opacity,
					classnames=self.classnames,
					save_path=os.path.join(self.output_dir, str(z).zfill(3)+"."+self.image_format))
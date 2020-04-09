import os
import SimpleITK as sitk
from tqdm import tqdm

SRC_DIR = "/home/jacky/Projects/vnet-tensorflow/data_lits/testing"
TGT_DIR = "/mnt/data_disk/lits_crop/testing"
SELECT_LABEL = [1,2]
SRC_IMG_NAME = "image.nii"
SRC_LABEL_NAME = "label.nii"
TGT_IMG_NAME = "image_cropped.nii.gz"
TGT_LABEL_NAME = "label_cropped.nii.gz"
BUFFER = 2
MASK_IMAGE = True
MASK_DILATION = 2
MASK_DIM = [0,1,2]

def mask(image,label):
	dilateFilter = sitk.BinaryDilateImageFilter()
	dilateFilter.SetKernelRadius(MASK_DILATION)
	label = dilateFilter.Execute(label)

	label.SetSpacing(image.GetSpacing())
	label.SetOrigin(image.GetOrigin())
	label.SetDirection(image.GetDirection())

	maskFilter = sitk.MaskImageFilter()
	castFilter = sitk.CastImageFilter()
	# castFilter.SetOutputPixelType(sitk.sitkFloat32)
	# image = castFilter.Execute(image)
	# label = castFilter.Execute(label)
	image = maskFilter.Execute(image,label)
	return image

def main():
	binaryFilter = sitk.BinaryThresholdImageFilter()
	binaryFilter.SetOutsideValue(0)
	binaryFilter.SetInsideValue(1)

	reader = sitk.ImageFileReader()
	writer = sitk.ImageFileWriter()

	pbar = tqdm(os.listdir(SRC_DIR))

	for case in pbar:
		if not (os.path.isdir(os.path.join(SRC_DIR,case))):
			continue
		if not (os.path.exists(os.path.join(SRC_DIR,case,SRC_LABEL_NAME)) or os.path.exists(os.path.join(SRC_DIR,case,SRC_IMG_NAME))):
			continue

		pbar.set_description(case)
		reader.SetFileName(os.path.join(SRC_DIR,case,SRC_LABEL_NAME))
		label = reader.Execute()
		reader.SetFileName(os.path.join(SRC_DIR,case,SRC_IMG_NAME))
		image = reader.Execute()

		label_ = sitk.Image(label.GetSize(),sitk.sitkUInt8)
		label_.SetSpacing(label.GetSpacing())
		label_.SetDirection(label.GetDirection())
		label_.SetOrigin(label.GetOrigin())

		for i in SELECT_LABEL:
			binaryFilter.SetLowerThreshold(i)
			binaryFilter.SetUpperThreshold(i) 
			addFilter = sitk.AddImageFilter()
			label_ = addFilter.Execute(label_,binaryFilter.Execute(label))

		if MASK_IMAGE:
			image = mask(image,label_)

		# find bbox of label
		labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		labelShapeFilter.Execute(label_)
		bbox = labelShapeFilter.GetBoundingBox(1)

		if 0 in MASK_DIM:
			i_start = bbox[0] - BUFFER
			i_end = bbox[0] + bbox[3] + BUFFER
			if i_start < 0:
				i_start = 0

			if i_end >= label.GetSize()[0]:
				i_end = label.GetSize()[0] - 1
		else:
			i_start = 0
			i_end = label.GetSize()[0] - 1

		if 1 in MASK_DIM:
			j_start = bbox[1] - BUFFER
			j_end = bbox[1] + bbox[4] + BUFFER
			if j_start < 0:
				j_start = 0

			if j_end >= label.GetSize()[1]:
				j_end = label.GetSize()[1] - 1
		else:
			j_start = 0
			j_end = label.GetSize()[1] - 1

		if 2 in MASK_DIM:
			k_start = bbox[2] - BUFFER
			k_end = bbox[2] + bbox[5] + BUFFER

			if k_start < 0:
				k_start = 0

			if k_end >= label.GetSize()[2]:
				k_end = label.GetSize()[2] - 1
		else:
			k_start = 0
			k_end = label.GetSize()[2] - 1

		extractor = sitk.RegionOfInterestImageFilter()
		size = [i_end-i_start,j_end-j_start,k_end-k_start]
		index = [i_start,j_start,k_start]

		extractor.SetSize(size)
		extractor.SetIndex(index)
		image_cropped = extractor.Execute(image)
		label_cropped = extractor.Execute(label)

		os.makedirs(os.path.join(TGT_DIR,case),exist_ok=True)
		writer.SetFileName(os.path.join(TGT_DIR,case,TGT_LABEL_NAME))
		writer.Execute(label_cropped)
		writer.SetFileName(os.path.join(TGT_DIR,case,TGT_IMG_NAME))
		writer.Execute(image_cropped)

if __name__=="__main__":
	main()
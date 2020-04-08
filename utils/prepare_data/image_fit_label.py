import os
import SimpleITK as sitk
from tqdm import tqdm

SRC_DIR = "../../data_lits/training"
TGT_DIR = "/mnt/data_disk/lits_crop/training"
SELECT_LABEL = [1,2]
SRC_IMG_NAME = "image.nii"
SRC_LABEL_NAME = "label.nii"
TGT_IMG_NAME = "image_cropped.nii"
TGT_LABEL_NAME = "label_cropped.nii"
BUFFER = 2
MASK_IMAGE = True
MASK_DILATION = 2

def mask(image,label):
	dilateFilter = sitk.BinaryDilateImageFilter()
	dilateFilter.SetKernelRadius(MASK_DILATION)
	label = dilateFilter.Execute(label)

	label.SetSpacing(image.GetSpacing())
	label.SetOrigin(image.GetOrigin())
	label.SetDirection(image.GetDirection())

	maskFilter = sitk.MaskNegatedImageFilter()
	castFilter = sitk.CastImageFilter()
	castFilter.SetOutputPixelType(sitk.sitkFloat32)
	image = castFilter.Execute(image)
	label = castFilter.Execute(label)
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

		k_start = bbox[2] - BUFFER
		k_end = bbox[2] + bbox[5] + BUFFER

		if k_start < 0:
			k_start = 0

		if k_end >= label.GetSize()[2]:
			k_end = label.GetSize()[2] - 1

		extractor = sitk.RegionOfInterestImageFilter()
		size = [label.GetSize()[0],label.GetSize()[1],k_end-k_start]
		index = [0,0,k_start]

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
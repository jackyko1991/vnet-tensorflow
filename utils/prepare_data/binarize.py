import os
import SimpleITK as sitk
from tqdm import tqdm

SRC_DIR = "/mnt/data_disk/lits_crop/testing"
TGT_DIR = "/mnt/data_disk/lits_crop/testing"
SELECT_LABEL = [2]
MASK_LABEL = []
# MASK_LABEL = [1,2]
SRC_IMG_NAME = "image_cropped.nii.gz"
SRC_LABEL_NAME = "label_cropped.nii.gz"
MASKED_IMG_NAME = "image_masked.nii.gz"
TGT_LABEL_NAME = "label_masked.nii.gz"
MASK_DILATION = 5

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

		label_out = sitk.Image(label.GetSize(),sitk.sitkUInt8)
		label_out.SetSpacing(label.GetSpacing())
		label_out.SetDirection(label.GetDirection())
		label_out.SetOrigin(label.GetOrigin())

		for i in SELECT_LABEL:
			binaryFilter.SetLowerThreshold(i)
			binaryFilter.SetUpperThreshold(i)
			label_ = binaryFilter.Execute(label)
			addFilter = sitk.AddImageFilter()
			label_out = addFilter.Execute(label_out,label_)

		os.makedirs(os.path.join(TGT_DIR,case),exist_ok=True)
		writer.SetFileName(os.path.join(TGT_DIR,case,TGT_LABEL_NAME))
		writer.Execute(label_out)

		if len(MASK_LABEL) > 0:
			reader.SetFileName(os.path.join(SRC_DIR,case,SRC_IMG_NAME))
			image = reader.Execute()

			mask = sitk.Image(label.GetSize(),sitk.sitkUInt8)
			mask.SetSpacing(label.GetSpacing())
			mask.SetDirection(label.GetDirection())
			mask.SetOrigin(label.GetOrigin())

			for i in MASK_LABEL:
				binaryFilter.SetLowerThreshold(i)
				binaryFilter.SetUpperThreshold(i)
				mask_ = binaryFilter.Execute(label)
				addFilter = sitk.AddImageFilter()
				mask = addFilter.Execute(mask,mask_)

			if MASK_DILATION > 0:
				dilateFilter = sitk.BinaryDilateImageFilter()
				dilateFilter.SetKernelRadius(MASK_DILATION)
				mask = dilateFilter.Execute(mask)

			maskFilter = sitk.MaskImageFilter()
			image_masked = maskFilter.Execute(image, mask)

			writer.SetFileName(os.path.join(TGT_DIR,case,MASKED_IMG_NAME))
			writer.Execute(image_masked)

if __name__=="__main__":
	main()
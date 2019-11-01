import os
import SimpleITK as sitk
from tqdm import tqdm
import math

SRC_DIR = "../../data_lits/training"
TGT_DIR = "../../data_lits_partition/training"
SRC_IMG_NAME = "image_cropped.nii"
SRC_LABEL_NAME = "label_cropped.nii"
TGT_IMG_NAME = "image.nii.gz"
TGT_LABEL_NAME = "label.nii.gz"
LAYER = 64

def main():
	extractor = sitk.RegionOfInterestImageFilter()

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
		
		for k_start in range(0,label.GetSize()[2],LAYER):
			index = [0,0,k_start]
			if (k_start + LAYER) < label.GetSize()[2]:
				size = [label.GetSize()[0],label.GetSize()[1],LAYER]
			else:
				size = [label.GetSize()[0],label.GetSize()[1],label.GetSize()[2]-k_start]

			extractor.SetSize(size)
			extractor.SetIndex(index)
			image_cropped = extractor.Execute(image)
			label_cropped = extractor.Execute(label)

			os.makedirs(os.path.join(TGT_DIR,case+ "_"+str(k_start)),exist_ok=True)
			writer.SetFileName(os.path.join(TGT_DIR,case+ "_"+str(k_start),TGT_LABEL_NAME))
			writer.Execute(label_cropped)
			writer.SetFileName(os.path.join(TGT_DIR,case+ "_" +str(k_start),TGT_IMG_NAME))
			writer.Execute(image_cropped)

if __name__=="__main__":
	main()
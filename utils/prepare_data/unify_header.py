import os
import SimpleITK as sitk
from tqdm import tqdm

def main():
	data_dir = "/mnt/data_disk/ADAM_release_subjs/testing"

	pbar = tqdm(os.listdir(data_dir))
	image_1_filename = "TOF.nii.gz"
	image_2_filename = "struct_aligned.nii.gz"

	for case in pbar:
		pbar.set_description(case)
		reader = sitk.ImageFileReader()
		reader.SetFileName(os.path.join(data_dir,case,image_1_filename))
		image1 = reader.Execute()
		reader.SetFileName(os.path.join(data_dir,case,image_2_filename))
		image2 = reader.Execute()

		if (image1.GetDirection() == image2.GetDirection()):
			continue

		image1_ = sitk.GetImageFromArray(sitk.GetArrayFromImage(image1))
		image2_ = sitk.GetImageFromArray(sitk.GetArrayFromImage(image2))

		image1_.SetOrigin(image1.GetOrigin())
		image1_.SetDirection(image1.GetDirection())
		image2_.SetOrigin(image1.GetOrigin())
		image2_.SetDirection(image1.GetDirection())

		writer = sitk.ImageFileWriter()

		os.remove(os.path.join(data_dir,case,image_1_filename))
		writer.SetFileName(os.path.join(data_dir,case,image_1_filename))
		writer.Execute(image1_)

		os.remove(os.path.join(data_dir,case,image_2_filename))

		writer.SetFileName(os.path.join(data_dir,case,image_2_filename))
		writer.Execute(image2_)

if __name__=="__main__":
	main()
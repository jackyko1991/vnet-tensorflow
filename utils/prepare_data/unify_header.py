import os
import SimpleITK as sitk
from tqdm import tqdm

def main():
	data_dir = "/users/kir-fritzsche/oyk357/archive/cow_data/split/fold_0/testing"

	# pbar = tqdm(os.listdir(data_dir))
	pbar = tqdm(["topcow_mr_007","topcow_mr_013","topcow_mr_016","topcow_mr_017"])
	image_1_filename = "image.nii.gz"
	image_2_filename = "label.nii.gz"

	for case in pbar:
		pbar.set_description(case)
		image1_path = os.path.join(data_dir,case,image_1_filename)
		image2_path = os.path.join(data_dir,case,image_2_filename)

		if not (os.path.exists(image1_path) and os.path.exists(image2_path)):
			continue

		reader = sitk.ImageFileReader()
		reader.SetFileName(image1_path)
		image1 = reader.Execute()
		reader.SetFileName(image2_path)
		image2 = reader.Execute()

		image1_ = sitk.GetImageFromArray(sitk.GetArrayFromImage(image1))
		image2_ = sitk.GetImageFromArray(sitk.GetArrayFromImage(image2))

		image1_.SetOrigin(image1.GetOrigin())
		image1_.SetDirection(image1.GetDirection())
		image1_.SetSpacing(image1.GetSpacing())
		image2_.SetOrigin(image1.GetOrigin())
		image2_.SetDirection(image1.GetDirection())
		image2_.SetSpacing(image1.GetSpacing())
		# image1_.CopyInformation(image1)
		# image2_.CopyInformation(image1)

		writer = sitk.ImageFileWriter()

		os.remove(os.path.join(data_dir,case,image_1_filename))
		writer.SetFileName(os.path.join(data_dir,case,image_1_filename))
		writer.Execute(image1_)

		os.remove(os.path.join(data_dir,case,image_2_filename))

		writer.SetFileName(os.path.join(data_dir,case,image_2_filename))
		writer.Execute(image2_)

if __name__=="__main__":
	main()
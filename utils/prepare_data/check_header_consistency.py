import os
import SimpleITK as sitk
from tqdm import tqdm

def main():
	data_dir = "/users/kir-fritzsche/oyk357/archive/cow_data/split/fold_0/testing"

	pbar = tqdm(os.listdir(data_dir))
	# pbar = tqdm(["topcow_mr_007","topcow_mr_013","topcow_mr_016","topcow_mr_017"])
	for case in pbar:
		pbar.set_description(case)
		reader = sitk.ImageFileReader()
		image_path = os.path.join(data_dir,case,"image.nii.gz")
		label_path = os.path.join(data_dir,case,"label.nii.gz")

		if not (os.path.exists(image_path) and os.path.exists(label_path)):
			continue

		reader.SetFileName(image_path)
		image = reader.Execute()
		reader.SetFileName(label_path)
		label = reader.Execute()

		if not (image.GetSize() == label.GetSize()):
			print(case, "size")
			print(image.GetSize())
			print(label.GetSize())
		if not (image.GetDirection() == label.GetDirection()):
			print(case, "direction")
			print(image.GetDirection())
			print(label.GetDirection())
		if not (image.GetOrigin() == label.GetOrigin()):
			print(case, "origin")
			print(image.GetOrigin())
			print(label.GetOrigin())
		if not (image.GetSpacing() == label.GetSpacing()):
			print(case, "spacing")
			print(image.GetSpacing())
			print(label.GetSpacing())

if __name__=="__main__":
	main()
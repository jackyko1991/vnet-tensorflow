import os
import SimpleITK as sitk
from tqdm import tqdm

def main():
	data_dir = "/mnt/data_disk/ADAM_release_subjs/training"

	pbar = tqdm(os.listdir(data_dir))

	for case in pbar:
		pbar.set_description(case)
		reader = sitk.ImageFileReader()
		reader.SetFileName(os.path.join(data_dir,case,"TOF.nii.gz"))
		image = reader.Execute()
		reader.SetFileName(os.path.join(data_dir,case,"aneurysms.nii.gz"))
		label = reader.Execute()

		if (image.GetDirection() == label.GetDirection()):
			continue

		label.SetDirection(image.GetDirection())

		writer = sitk.ImageFileWriter()
		writer.SetFileName(os.path.join(data_dir,case,"TOF.nii.gz"))
		writer.Execute(image)
		writer.SetFileName(os.path.join(data_dir,case,"aneurysms.nii.gz"))
		writer.Execute(label)

if __name__=="__main__":
	main()
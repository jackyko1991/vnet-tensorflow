import os
import SimpleITK as sitk

def main():
	data_dir = "/mnt/data_disk/ADAM_release_subjs/training"

	for case in os.listdir(data_dir):
		reader = sitk.ImageFileReader()
		reader.SetFileName(os.path.join(data_dir,case,"TOF.nii.gz"))
		image = reader.Execute()
		reader.SetFileName(os.path.join(data_dir,case,"aneurysms.nii.gz"))
		label = reader.Execute()

		if not (image.GetSize() == label.GetSize()):
			print(case, "size")
		if not (image.GetDirection() == label.GetDirection()):
			print(case, "direction")
		if not (image.GetOrigin() == label.GetOrigin()):
			print(case, "origin")


if __name__=="__main__":
	main()
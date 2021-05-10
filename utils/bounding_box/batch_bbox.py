from bbox import *
from tqdm import tqdm

def main():
	data_dir ="Z:\\data\\intracranial_hemorrhage_segmentation\\combined_dataset\\evaluate"
	output_dir_name = "ground_truth"
	# classname_file = "Z:\\data\\intracranial_hemorrhage_segmentation\\combined_dataset\\classnames.json"
	classname_file = ""

	pbar = tqdm(os.listdir(data_dir))
	for case in pbar:
		pbar.set_description(case)

		image_path = os.path.join(data_dir,case,"image.nii.gz")
		label_path = os.path.join(data_dir,case,"label.nii.gz")
		output_dir = os.path.join(data_dir,case,output_dir_name)

		if not (os.path.exists(image_path) and os.path.exists(label_path)):
			continue

		bbox = BoundingBox(image_path, label_path, output_dir)
		bbox.min_intensity = 0
		bbox.max_intensity = 80
		bbox.opacity = 0.7
		bbox.classname_file = classname_file
		bbox.run()

if __name__ == "__main__":
	main()
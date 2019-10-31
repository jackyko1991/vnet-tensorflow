import os
from tqdm import tqdm

SRC_DIR = "../../data_lits"
TGT_DIR = "../../data_lits"

def main():
	pbar = tqdm(os.listdir(SRC_DIR))

	for file in pbar:
		if not ".nii" in file:
			continue
		case = ''.join([str(s) for s in file if s.isdigit()])

		src = os.path.join(SRC_DIR, file)

		os.makedirs(os.path.join(TGT_DIR,case),exist_ok=True)
		if 'volume' in file:
			if '.nii.gz' in file:
				tgt = os.path.join(TGT_DIR, case, 'image.nii.gz')
			else:
				tgt = os.path.join(TGT_DIR, case, 'image.nii')
		elif 'segmentation' in file:
			if '.nii.gz' in file:
				tgt = os.path.join(TGT_DIR, case, 'label.nii.gz')
			else:
				tgt = os.path.join(TGT_DIR, case, 'label.nii')

		os.rename(src,tgt)
		pbar.set_description("Moving data: {}".format(file))

if __name__=="__main__":
	main()
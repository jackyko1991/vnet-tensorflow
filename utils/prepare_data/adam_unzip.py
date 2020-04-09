import os
from zipfile import ZipFile
from tqdm import tqdm
import shutil

def zipCopy(zipObj,src,dst):
	with zipObj.open(src) as zf, open(dst, 'wb') as f:
		shutil.copyfileobj(zf, f)

def unzip(file, dest):
	if not os.path.exists(dest):
		os.makedirs(dest)

	with ZipFile(file,'r') as zipObj:
		filenameList = zipObj.namelist()
		case = filenameList[0]
		struct_src = os.path.join(case,'pre','struct_aligned.nii.gz')
		tof_src = os.path.join(case,'pre','TOF.nii.gz')
		aneurysms_src = os.path.join(case,'aneurysms.nii.gz')
		location_src = os.path.join(case,'location.txt')

		struct_tgt = os.path.join(dest,'struct_aligned.nii.gz')
		tof_tgt = os.path.join(dest,'TOF.nii.gz')
		aneurysms_tgt = os.path.join(dest,'aneurysms.nii.gz')
		location_tgt = os.path.join(dest,'location.txt')

		zipCopy(zipObj,struct_src,struct_tgt)
		zipCopy(zipObj,tof_src,tof_tgt)
		zipCopy(zipObj,aneurysms_src,aneurysms_tgt)
		zipCopy(zipObj,location_src,location_tgt)

def main():
	src_dir = "/mnt/data_disk/ADAM_release_subjs_raw"
	tgt_dir = "/mnt/data_disk/ADAM_release_subjs"

	pbar = tqdm(os.listdir(src_dir))

	for filename in pbar:
		pbar.set_description(filename)
		unzip(os.path.join(src_dir,filename),os.path.join(tgt_dir,filename[:-4]))

if __name__ == "__main__":
	main()
import os
import SimpleITK as sitk
import tensorflow as tf

class Batch_Evaluate:
	def __init__(self,
		model_folder="./tmp/ckpt",
		output_folder ="./tmp",
		data_folder = "./data",
		stride_min=30,
		stride_max=64,
		step=2,
		checkpoint_min=70000,
		checkpoint_max=9999999999999999999999999,
		batch_size=10):
		self.model_folder = model_folder
		self.output_folder = output_folder
		self.data_folder = data_folder

		assert isinstance(step, int)
		assert step > 0
		self.stride_min = stride_min
		self.stride_max = stride_max

		assert isinstance(step, int)
		assert step > 0
		self.step = step

		assert isinstance(checkpoint_min, int)
		assert checkpoint_min > 0
		self.checkpoint_min = checkpoint_min
		assert isinstance(checkpoint_max, int)
		assert checkpoint_max > 0
		self.checkpoint_max = checkpoint_max

	@property
	def model_folder(self):
		return self._model_folder
	
	@model_folder.setter
	def model_folder(self, model_folder):
		assert isinstance(model_folder, str)
		self._model_folder=model_folder

	@property
	def output_folder(self):
		return self._output_folder

	@output_folder.setter
	def output_folder(self, output_folder):
		assert isinstance(output_folder, str)
		self._output_folder=output_folder
		self._output_csv_path = self._output_folder + "result.csv"

	@property
	def data_folder(self):
		return self._data_folder

	@data_folder.setter
	def data_folder(self, data_folder):
		assert isinstance(data_folder, str)
		self._data_folder = data_folder

	def Execute(self):
		ckpts = os.listdir(self._model_folder)
		ckpts =  [x for x in ckpts if '.meta' in x]

		if not os.path.exists(self._output_folder):
			os.makedirs(self._output_folder)

		print(ckpts)

		for ckpt in ckpts:
			model_path = os.path.join(self._model_folder,ckpt)
			checkpoint_path = os.path.join(ckpt_dir,ckpt.split(".")[0])
			checkpoint_num = int(ckpt.split(".")[0].split("-")[1])

			if checkpoint_num < self.checkpoint_min or checkpoint_num > self.checkpoint_max:
				continue

			for stride in range(self.stride_min,self.stride_max+1, self.step):
				print("Evaluation with stride {}".format(stride))

				command = "python ../../evaluate.py " + \
					"--data_dir " + self._data_folder + " " + \
					"--model_path " + model_path +  " " +\
					"--checkpoint_path " + checkpoint_path +  " " +\
					"--batch_size " + str(10) + " " +\
					"--stride_inplane " + str(stride) + " " +\
					"--stride_layer " + str(stride)

				print(command)
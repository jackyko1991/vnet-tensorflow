import os
import SimpleITK as sitk
import tensorflow as tf
import csv
import numpy as np
import subprocess
from tqdm import tqdm

def dist(a,b):
	x = 0
	for i in range(3):
		x += (a[i]-b[i])**2
	return math.sqrt(x)

def overlapMeasure(imageA, imageB, method="dice"):
	if not (method == "dice" or method == "jaccard"):
		print("invalid method")
		return 0

	overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
	overlapFilter.Execute(imageA, imageB)

	if method == "dice":
		return overlapFilter.GetDiceCoefficient()
	elif method == "jaccard":
		return overlapFilter.GetJaccardCoefficient()
	else:
		return 0

def Accuracy(groundTruth,output, tolerence=3, mode=None):
	# cast labels to int
	castFilter = sitk.CastImageFilter()
	castFilter.SetOutputPixelType(sitk.sitkUInt16)
	groundTruth = castFilter.Execute(groundTruth)
	output = castFilter.Execute(output)
	output.SetOrigin(groundTruth.GetOrigin())
	output.SetDirection(groundTruth.GetDirection())
	output.SetSpacing(groundTruth.GetSpacing())

	result = {}

	if 'DICE' in mode:
		# dice and jaccard
		result['DICE'] = overlapMeasure(groundTruth, output,"dice")
		result['Jaccard'] = overlapMeasure(groundTruth, output,"jaccard")
	if 'ITEM' in mode:
		# locate ground truth location
		ccFilter = sitk.ConnectedComponentImageFilter()
		groundTruth = ccFilter.Execute(groundTruth)
		
		gtLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		gtLabelShapeFilter.Execute(groundTruth)
		gtCentroids = []
		for i in range(gtLabelShapeFilter.GetNumberOfLabels()):
			# if gtLabelShapeFilter.GetPhysicalSize(i+1) >= math.pi*(1.5)**3*4/3:
				gtCentroids.append(gtLabelShapeFilter.GetCentroid(i+1))

		# locate the label centroid from output file
		output = ccFilter.Execute(output)
		outputLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
		outputLabelShapeFilter.Execute(output)
		outputCentroids = []
		# for i in range(outputLabelShapeFilter.GetNumberOfLabels()):
		# 	# if outputLabelShapeFilter.GetPhysicalSize(i+1) >= math.pi*(1.5)**3*4/3:
		# 	if labelShapeFilter.GetBoundingBox(i+1)[5] >= 6:
		# 		outputCentroids.append(outputLabelShapeFilter.GetCentroid(i+1))

		thicknessThreshold = 6

		for i in range(outputLabelShapeFilter.GetNumberOfLabels()):
			if outputLabelShapeFilter.GetBoundingBox(i+1)[5] < thicknessThreshold or \
				outputLabelShapeFilter.GetBoundingBox(i+1)[3] <	2 or \
				outputLabelShapeFilter.GetBoundingBox(i+1)[4] <	2:
				continue
			else:
				outputCentroids.append(outputLabelShapeFilter.GetCentroid(i+1))
			# if outputLabelShapeFilter.GetPhysicalSize(i+1) >= math.pi*(1)**3*4/3:
			# 	outputCentroids.append(outputLabelShapeFilter.GetCentroid(i+1))
			# 	if outputLabelShapeFilter.GetPhysicalSize(i+1) < math.pi*(2.5)**3*4/3:
			# 		label_vol_0 += outputLabelShapeFilter.GetPhysicalSize(i+1)
			# 	else:
			# 		label_vol_1 += outputLabelShapeFilter.GetPhysicalSize(i+1)

		# handle no label cases
		if len(gtCentroids) == 0:
			return 0, len(outputCentroids), 0, 0, 0

		TP = 0
		FN = 0
		# search for true positive and false negative
		for gtCentroid in gtCentroids:
			# print("ground truth centroid:", gtCentroid)
			TP_found = False
			for outputCentroid in outputCentroids:
				if dist(gtCentroid, outputCentroid) < tolerence:
					TP_found = True
				else:
					continue
			if TP_found:
				TP += 1
			else:
				# print("this is a false negative:",gtCentroid)
				FN += 1

		sensitivity = TP/(TP+FN)
		iouScore = TP/(TP+len(outputCentroids)-TP+FN)
		FP = len(outputCentroids)-TP

		print("TP:",TP)
		print("FP:",FP)
		print("TN:",0)
		print("FN:",FN)
		print("Sensitivity:", sensitivity)
		print("IoU:", iouScore)
		print("DICE:", dice)
		print("Jaccard:", jaccard)
		return TP, FP, FN, dice, jaccard

	return result

class Batch_Evaluate:
	def __init__(self,
		config_json="./config.json",
		model_folder="./tmp/ckpt",
		output_folder ="./tmp",
		data_folder = "./data",
		ground_truth_filename = "label.nii.gz",
		evaluated_filename = "label_vnet.nii.gz",
		stride_layer_min=32,
		stride_layer_max=64,
		stride_inplane_min=32,
		stride_inplane_max=64,
		patch_size=64,
		patch_layer=64,
		step=2,
		checkpoint_min=1,
		checkpoint_max=9999999999999999999999999,
		batch_size=5,
		mode=["DICE"]):
		self.config_json = config_json
		self.model_folder = model_folder
		self.output_folder = output_folder
		self.data_folder = data_folder

		assert isinstance(ground_truth_filename, str)
		self.ground_truth_filename = ground_truth_filename
		assert isinstance(evaluated_filename, str)
		self.evaluated_filename = evaluated_filename

		assert isinstance(stride_layer_min, int)
		assert isinstance(stride_layer_max, int)
		assert stride_layer_min > 0
		assert stride_layer_max > 0
		self.stride_layer_min = stride_layer_min
		self.stride_layer_max = stride_layer_max

		assert isinstance(stride_inplane_min, int)
		assert isinstance(stride_inplane_max, int)
		assert stride_inplane_min > 0
		assert stride_inplane_max > 0
		self.stride_inplane_min = stride_inplane_min
		self.stride_inplane_max = stride_inplane_max

		assert isinstance(step, int)
		assert step > 0
		self.step = step

		assert isinstance(checkpoint_min, int)
		assert checkpoint_min > 0
		self.checkpoint_min = checkpoint_min
		assert isinstance(checkpoint_max, int)
		assert checkpoint_max > 0
		self.checkpoint_max = checkpoint_max

		self.batch_size = batch_size
		self.mode = mode # DICE, ITEM

	@property
	def model_folder(self):
		return self._model_folder
	
	@model_folder.setter
	def model_folder(self, model_folder):
		assert isinstance(model_folder, str)

		self._model_folder=os.path.abspath(model_folder)

	@property
	def output_folder(self):
		return self._output_folder

	@output_folder.setter
	def output_folder(self, output_folder):
		assert isinstance(output_folder, str)
		self._output_folder=os.path.abspath(output_folder)

	@property
	def data_folder(self):
		return self._data_folder

	@data_folder.setter
	def data_folder(self, data_folder):
		assert isinstance(data_folder, str)
		self._data_folder = os.path.abspath(data_folder)

	def Execute(self):
		ckpts = os.listdir(self._model_folder)
		ckpts =  [x for x in ckpts if '.meta' in x]

		if not os.path.exists(self._output_folder):
			os.makedirs(self._output_folder)

		# print(ckpts)

		max_dice = 0
		max_jaccard = 0

		best_dice_result = {"ckpt": ckpts[0], "stride_inplane": self.stride_inplane_min, "stride_layer": self.stride_layer_min}
		best_jaccard_result = {"ckpt": ckpts[0], "stride_inplane": self.stride_inplane_min, "stride_layer": self.stride_layer_min}

		pbar = tqdm(ckpts)
		for ckpt in pbar:
			pbar.set_description(ckpt)

			model_path = os.path.join(self._model_folder,ckpt)
			checkpoint_path = os.path.join(self._model_folder,ckpt.split(".")[0])
			checkpoint_num = int(ckpt.split(".")[0].split("-")[1])

			if checkpoint_num < self.checkpoint_min or checkpoint_num > self.checkpoint_max:
				continue

			for stride_inplane in range(self.stride_inplane_min,self.stride_inplane_max+1, self.step):
				for stride_layer in range(self.stride_layer_min,self.stride_layer_max+1, self.step):
					command = "python evaluate.py " + \
						"--data_dir " + self._data_folder + " " + \
						"--config_json " + self.config_json + " " + \
						"--model_path " + model_path +  " " +\
						"--checkpoint_path " + checkpoint_path +  " " +\
						"--batch_size " + str(self.batch_size) + " " +\
						"--stride_inplane " + str(stride_inplane) + " " +\
						"--stride_layer " + str(stride_layer) + " " + \
						"--patch_size " + str(self.patch_size) + " " + \
						"--patch_layer " + str(self.patch_layer)

					os.system(command)

					# create csv file for logging
					output_csv_path = os.path.join(self._output_folder, "result_checkpoint-" + str(checkpoint_num) + "_stride_inplane-" + str(stride_inplane) + "_stride_layer-" + str(stride_layer) + ".csv")

					if os.path.exists(output_csv_path):
						os.remove(output_csv_path)
					csvfile = open(output_csv_path,'w')
					filenames = ['Case']
					if 'DICE' in self.mode:
						filenames.append('DICE')
						filenames.append('Jaccard')
					if 'ITEM' in self.mode:
						filenames.append('TP')
						filenames.append('FP')
						filenames.append('FN')
						filenames.append('Item Sensitivity')
						filenames.append('Item IoU')
					filewriter = csv.DictWriter(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, fieldnames=filenames)
					filewriter.writeheader()

					# accuracy checking for evaluated models
					TP = []
					FP = []
					FN = []
					DICE = []
					Jaccard = []

					pbar_case = tqdm(os.listdir(self._data_folder))
					for case in pbar_case:
						pbar_case.set_description(case)
						if not os.path.exists(os.path.join(self._data_folder, case, self.ground_truth_filename)):
							continue
						if not os.path.exists(os.path.join(self._data_folder, case, self.evaluated_filename)):
							continue

						# read image first
						if 'DICE' in self.mode or "DICE" in self.mode or \
							'ITEM' in self.mode or "ITEM" in self.mode:
							reader = sitk.ImageFileReader()
							reader.SetFileName(os.path.join(self._data_folder, case, self.ground_truth_filename))
							groundTruth = reader.Execute()
							reader.SetFileName(os.path.join(self._data_folder, case, self.evaluated_filename))
							evaluated = reader.Execute()

						result = Accuracy(groundTruth, evaluated, mode=self.mode)
						result['Case'] = case
						if "DICE" in self.mode:
							tqdm.write("Case: {}, DICE: {}, Jaccard: {}".format(case,result["DICE"],result["Jaccard"]))
						filewriter.writerow(result)

						if 'DICE' in self.mode or "DICE" in self.mode:
							DICE.append(result['DICE'])
							Jaccard.append(result['Jaccard'])

						exit()

					avg_result = {'Case': "average", 
						'DICE': np.sum(DICE)/len(DICE), 
						'Jaccard': np.sum(Jaccard)/len(Jaccard)}
					filewriter.writerow(avg_result)	
					csvfile.close()

					if avg_result["DICE"] > max_dice:
						max_dice = avg_result["DICE"]
						best_dice_result = {"ckpt": ckpt, "stride_inplane": stride_inplane, "stride_layer": stride_layer}

					if avg_result["Jaccard"] > max_jaccard:
						max_jaccard = avg_result["Jaccard"]
						best_jaccard_result = {"ckpt": ckpt, "stride_inplane": stride_inplane, "stride_layer": stride_layer}

		print("Best DICE result:", best_dice_result)
		print("Best Jaccard result:", best_jaccard_result)
import os
import SimpleITK as sitk
import math
import csv

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

def Accuracy(gtName,outputName, tolerence=2, thicknessThreshold=6):
	reader = sitk.ImageFileReader()
	reader.SetFileName(gtName)
	groundTruth = reader.Execute()

	reader.SetFileName(outputName)
	output = reader.Execute()

	# cast label to int
	castFilter = sitk.CastImageFilter()
	castFilter.SetOutputPixelType(sitk.sitkUInt16)
	groundTruth = castFilter.Execute(groundTruth)
	output = castFilter.Execute(output)
	output.SetOrigin(groundTruth.GetOrigin())
	output.SetDirection(groundTruth.GetDirection())
	output.SetSpacing(groundTruth.GetSpacing())

	# dice and jaccard
	dice = overlapMeasure(groundTruth, output,"dice")
	jaccard = overlapMeasure(groundTruth, output,"jaccard")

	# locate ground truth location
	ccFilter = sitk.ConnectedComponentImageFilter()
	groundTruth = ccFilter.Execute(groundTruth)
	
	gtLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
	gtLabelShapeFilter.Execute(groundTruth)
	gtCentroids = []
	gt_vol_0 = 0
	gt_vol_1 = 0

	for i in range(gtLabelShapeFilter.GetNumberOfLabels()):
		# gtCentroids.append(gtLabelShapeFilter.GetCentroid(i+1))
		if gtLabelShapeFilter.GetPhysicalSize(i+1) >= math.pi*(1)**3*4/3:
			gtCentroids.append(gtLabelShapeFilter.GetCentroid(i+1))
			if gtLabelShapeFilter.GetPhysicalSize(i+1) < math.pi*(2.5)**3*4/3:
				gt_vol_0 += gtLabelShapeFilter.GetPhysicalSize(i+1)
			else:
				gt_vol_1 += gtLabelShapeFilter.GetPhysicalSize(i+1)

	# locate the label centroid from output file
	output = ccFilter.Execute(output)
	outputLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
	outputLabelShapeFilter.Execute(output)
	outputCentroids = []

	label_vol_0 = 0
	label_vol_1 = 0

	for i in range(outputLabelShapeFilter.GetNumberOfLabels()):
		if outputLabelShapeFilter.GetBoundingBox(i+1)[5] < thicknessThreshold:
			continue
		if outputLabelShapeFilter.GetPhysicalSize(i+1) >= math.pi*(1)**3*4/3:
			outputCentroids.append(outputLabelShapeFilter.GetCentroid(i+1))
			if outputLabelShapeFilter.GetPhysicalSize(i+1) < math.pi*(2.5)**3*4/3:
				label_vol_0 += outputLabelShapeFilter.GetPhysicalSize(i+1)
			else:
				label_vol_1 += outputLabelShapeFilter.GetPhysicalSize(i+1)

	# handle no label cases
	if len(gtCentroids) == 0:
		return 0, len(outputCentroids), 0, 0, 0, 0,0,0,0

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

	print(len(outputCentroids))
	print("TP:",TP)
	print("FP:",FP)
	print("TN:",0)
	print("FN:",FN)
	print("Sensitivity:", sensitivity)
	print("IoU:", iouScore)
	print("DICE:", dice)
	print("Jaccard:", jaccard)
	return TP, FP, FN, dice, jaccard, gt_vol_0, gt_vol_1, label_vol_0, label_vol_1

def main():
	model_path = "./tmp/ckpt/checkpoint-76245.meta"
	checkpoint_path = "./tmp/ckpt/checkpoint-76245"
	output_csv_folder = "./tmp"
	dataDir = "./data_SWAN/evaluate"

	max_stride = 64
	min_stride = 30
	step = 2

	for stride in range(min_stride,max_stride+1,step):
		print("Evaluation with stride {}".format(stride))
		output_csv_path = os.path.join(output_csv_folder,"result_stride_" + str(stride) + ".csv")

		if os.path.exists(output_csv_path):
			os.remove(output_csv_path)

		csvfile = open(output_csv_path, 'w')
		filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(['Case', 'TP', 'FP', 'FN', 'Item Sensitivity', 'Item IoU', 'DICE', 'Jaccard', 'GT_vol<5mm', 'GT_vol>5mm', 'VNet_vol<5mm', 'VNet_vol>5mm'])

		command = "python ./evaluate.py " + \
			"--data_dir " + dataDir + " " + \
			"--model_path " + model_path +  " " +\
			"--checkpoint_path " + checkpoint_path +  " " +\
			"--batch_size " + str(10) + " " +\
			"--stride_inplane " + str(stride) + " " +\
			"--stride_layer " + str(stride)

		os.system(command)

		# perform accuracy check
		TP = 0
		FP = 0
		FN = 0
		DICE = 0
		Jaccard = 0
		GT_vol_0 = 0
		GT_vol_1 = 0
		VNet_vol_0 = 0
		VNet_vol_1 = 0

		for case in os.listdir(dataDir):
			if not os.path.exists(os.path.join(dataDir,case,"label_vnet.nii.gz")):
				continue
			print("case:",case)
			_TP, _FP, _FN, _DICE, _Jaccard, _GT_vol_0, _GT_vol_1, _VNet_vol_0, _VNet_vol_1 = \
				Accuracy(os.path.join(dataDir,case,"label_crop.nii.gz"),os.path.join(dataDir,case,"label_vnet.nii.gz"))
			TP += _TP
			FP += _FP
			FN += _FN
			DICE += _DICE
			Jaccard += _Jaccard
			GT_vol_0 += _GT_vol_0
			GT_vol_1 += _GT_vol_1
			VNet_vol_0 += _VNet_vol_0
			VNet_vol_1 += _VNet_vol_1

			if (_TP+_FN) == 0:
				_sensitivity = "nan"
			else:
				_sensitivity = TP/(TP+FN)
			if (_TP+_FP+_FN) == 0:
				_iou = "nan"
			else:
				_iou = _TP/(_TP+_FP+_FN)

			filewriter.writerow([case,_TP, _FP, _FN, _sensitivity, _iou, _DICE, _Jaccard,\
			_GT_vol_0, _GT_vol_1, _VNet_vol_0, _VNet_vol_1])

		print("Evaluation result of stride {}:".format(stride))
		if (TP+FN) == 0:
			avg_sensitivity = "nan"
		else:
			avg_sensitivity = TP/(TP+FN)
		if (TP+FP+FN) == 0:
			avg_iou = "nan"
		else:
			avg_iou = TP/(TP+FP+FN)

		print("Average Sensitivity:", avg_sensitivity)
		print("Average IoU:", avg_iou)

		filewriter.writerow(["result "+str(stride),TP, FP, FN, avg_sensitivity, avg_iou, DICE/len(os.listdir(dataDir)), Jaccard/len(os.listdir(dataDir)),\
		GT_vol_0, GT_vol_1, VNet_vol_0, VNet_vol_1])

	csvfile.close()

if __name__=="__main__":
	main()
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

def Accuracy(gtName,outputName, tolerence=3):
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
	for i in range(gtLabelShapeFilter.GetNumberOfLabels()):
		gtCentroids.append(gtLabelShapeFilter.GetCentroid(i+1))

	# locate the label centroid from output file
	output = ccFilter.Execute(output)
	outputLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
	outputLabelShapeFilter.Execute(output)
	outputCentroids = []
	for i in range(outputLabelShapeFilter.GetNumberOfLabels()):
		outputCentroids.append(outputLabelShapeFilter.GetCentroid(i+1))

	# handle no label cases
	if len(gtCentroids) == 0:
		return 0, len(outputCentroids), 0

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
	return TP, FP, FN, dice, jaccard

def main():
	ckpt_dir = "./tmp/ckpt"
	output_csv_dir = "./tmp/evalaution_result"
	dataDir = "./data_SWAN/evaluate"
	ckpts = os.listdir(ckpt_dir)

	ckpts =  [x for x in ckpts if '.meta' in x]

	if not os.path.exists(output_csv_dir):
		os.makedirs(output_csv_dir)

	for ckpt in ckpts:
		model_path = os.path.join(ckpt_dir,ckpt)
		checkpoint_path = os.path.join(ckpt_dir,ckpt.split(".")[0])
		checkpoint_num = int(ckpt.split(".")[0].split("-")[1])

		if checkpoint_num < 35000:
			continue
		command = "python evaluate.py " + \
			"--model_path " + model_path +  " " +\
			"--checkpoint_path " + checkpoint_path +  " " +\
			"--batch_size " + str(5)
		os.system(command)

		# perform accuracy check
		TP = 0
		FP = 0
		FN = 0
		DICE = 0
		Jaccard = 0

		if os.path.exists(os.path.join(output_csv_dir,str(checkpoint_num)+".csv")):
			os.remove(os.path.join(output_csv_dir,str(checkpoint_num)+".csv"))
		csvfile = open(os.path.join(output_csv_dir,str(checkpoint_num)+".csv"), 'w')
		filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(['Case', 'TP', 'FP', 'FN', 'Sensitivity', 'IoU', 'DICE', 'Jaccard'])

		for case in os.listdir(dataDir):
			if not os.path.exists(os.path.join(dataDir,case,"label_vnet.nii.gz")):
				continue
			print("case:",case)
			_TP, _FP, _FN, _DICE, _Jaccard = Accuracy(os.path.join(dataDir,case,"label_crop.nii.gz"),os.path.join(dataDir,case,"label_vnet.nii.gz"))
			TP += _TP
			FP += _FP
			FN += _FN
			DICE += _DICE
			Jaccard += _Jaccard

			if (_TP+_FN) == 0:
				_sensitivity = "nan"
			else:
				_sensitivity = TP/(TP+FN)
			if (_TP+_FP+_FN) == 0:
				_iou = "nan"
			else:
				_iou = _TP/(_TP+_FP+_FN)
			filewriter.writerow([case, _TP, _FP, _FN, _sensitivity, _iou, _DICE, _Jaccard])


		print("Evaluation result of checkpoint {}:".format(ckpt))
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

		filewriter.writerow(["average", TP, FP, FN, avg_sensitivity, avg_iou, DICE/len(os.listdir(dataDir)), Jaccard/len(os.listdir(dataDir))])
		csvfile.close()

if __name__=="__main__":
	main()

	
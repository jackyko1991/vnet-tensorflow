#include "tf_inference.h"

#include "iostream"
#include "ctime"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

int main() 
{
	// load inference image
	std::string imagePath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/done/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/image.nii.gz");
	//std::string imagePath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/test/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/image_crop.nii");

	// load tensorflow graph
	std::string graphPath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/graph.pb");
	std::string checkpointPath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/ckpt/checkpoint-67591.meta");

	using ImageReaderType = itk::ImageFileReader<ImageType>;

	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	imageReader->SetFileName(imagePath);
	imageReader->Update();

	clock_t cl;
	cl = clock();

	TF_Inference tf_Inference;
	tf_Inference.SetImage(imageReader->GetOutput());
	tf_Inference.SetCheckpointPath(checkpointPath);
	tf_Inference.SetGraphPath(graphPath);
	tf_Inference.Inference();

	cl = clock() - cl;

	std::cout << "Inferece time: " << cl/(double)CLOCKS_PER_SEC  << "s"<<std:: endl;  //prints the determined ticks per second (seconds passed)

	itk::ImageFileWriter<LabelImageType>::Pointer writer = itk::ImageFileWriter<LabelImageType>::New();
	writer->SetInput(tf_Inference.GetOutput());
	writer->SetFileName("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/test/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/label_final.nii.gz");
	writer->Write();

	system("pause");
}
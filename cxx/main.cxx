//#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/tensor.h"

#include "tf_inference.h"


//#include "tensorflow/core/public/session.h"
//#include "tensorflow/core/public/session_options.h"
//#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "iostream"

#include "itkImage.h"
#include "itkImageFileReader.h"

int main() 
{
	// load inference image
	//std::string imagePath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/done/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/image_windowed.nii.gz");
	std::string imagePath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/test/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/image_crop.nii");

	// load tensorflow graph
	std::string graphPath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/graph.pb");
	std::string checkpointPath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/ckpt/checkpoint-67591.meta");


	using ImageReaderType = itk::ImageFileReader<ImageType>;

	ImageReaderType::Pointer imageReader = ImageReaderType::New();
	imageReader->SetFileName(imagePath);
	imageReader->Update();

	TF_Inference tf_Inference;
	tf_Inference.SetImage(imageReader->GetOutput());
	tf_Inference.SetCheckpointPath(checkpointPath);
	tf_Inference.SetGraphPath(graphPath);
	tf_Inference.Inference();

	//system("pause");
}
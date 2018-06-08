//#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "iostream"

//#include "itkImage.h"
//#include "itkImageFileReader.h"
//#include "itkImageFileWriter.h"
//
//using ImageType = itk::Image<float, 3>;
//using LabelImageType = itk::Image<short, 3>;

int main() 
{
	//// load inference image
	//std::string imagePath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/evaluate/13028890657_20170730_2.16.840.114421.12234.9554736277.9586272277/image_windowed.nii.gz");
	//
	//using ImageReaderType = itk::ImageFileReader<ImageType>;
	//ImageReaderType::Pointer imageReader = ImageReaderType::New();
	//imageReader->SetFileName(imagePath);
	//imageReader->Update();

	// restore tensorflow graph
	std::string graphPath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/graph.pb");
	std::string checkpointPath = std::string("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/tmp/ckpt/checkpoint-67591.meta");

	// prepare tensorflow session
	tensorflow::Session* sess;
	tensorflow::SessionOptions options;
	TF_CHECK_OK(tensorflow::NewSession(options, &sess));
	std::cout << "Tensorflow session started" << std::endl;

	// load the graph proto file
	tensorflow::GraphDef graphDef;
	
	TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graphPath, &graphDef));
	std::cout << "Tensorflow load graph complete" << std::endl;

	// add graph to session
	TF_CHECK_OK(sess->Create(graphDef));
	std::cout << "Tensorflow add graph to session complete" << std::endl;

	// load weights to the graph
	std::vector<tensorflow::Tensor> out;
	std::vector<std::string> vNames;

	for (int i = 0; i < graphDef.node_size(); i++)
	{
		tensorflow::NodeDef n = graphDef.node(i);

		if (n.name().find("nWeights") != std::string::npos) {
			vNames.push_back(n.name());
			std::cout << n.name()<<std::endl;
		}
	}

	TF_CHECK_OK(sess->Run({}, vNames, {}, &out));
	std::cout << "Tensorflow load weight complete" << std::endl;

	//tensorflow::TensorShape inputShape;
	//inputShape.InsertDim(0, 1);
	//inputShape.InsertDim(1, 64);
	//inputShape.InsertDim(2, 64);
	//inputShape.InsertDim(3, 32);
	//inputShape.InsertDim(4, 1);

	tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1,64,64,32,1 }));
	
	auto inputTensorMapped = inputTensor.tensor<float, 5>();

	float count = 0;

	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			for (int k = 0; k < 32; k++)
			{
				inputTensorMapped(0, i, j, k, 0) = count;
				count++;
			}
		}
	}

	std::vector<std::pair<std::string, tensorflow::Tensor>> input;
	std::vector<tensorflow::Tensor> answer;

	input.emplace_back(std::string("images_placeholder:0"), inputTensor);

	auto statusPred = sess->Run(input, { "predicted_label/prediction:0" }, {}, &answer);

	sess->Close();
}
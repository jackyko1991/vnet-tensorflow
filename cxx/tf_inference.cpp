#include "tf_inference.h"

TF_Inference::TF_Inference()
{
	m_inputImage = ImageType::New();

	// tensorflow session
	TF_CHECK_OK(tensorflow::NewSession(m_options, &m_sess));

	// tensorflow graph
	m_graphDef = new tensorflow::GraphDef();
}

TF_Inference::~TF_Inference()
{
	// remove graphDef
	delete m_graphDef;

	// close tensorflow session
	m_sess->Close();
}

void TF_Inference::SetImage(ImageType::Pointer image)
{
	m_inputImage->Graft(image);
	m_inputImage->Print(std::cout);
}

void TF_Inference::SetGraphPath(std::string path)
{
	m_graphPath = path;
}

void TF_Inference::SetCheckpointPath(std::string path)
{
	m_checkpointPath = path;
}

void TF_Inference::Inference()
{
	// load the graph proto file
	TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), m_graphPath, m_graphDef));
	std::cout << "Tensorflow load graph complete" << std::endl;

	// add graph to session
	TF_CHECK_OK(m_sess->Create(*m_graphDef));
	std::cout << "Tensorflow add graph to session complete" << std::endl;

	// load weights to the graph
	std::vector<tensorflow::Tensor> out;
	std::vector<std::string> vNames;

	for (int i = 0; i < m_graphDef->node_size(); i++)
	{
		tensorflow::NodeDef n = m_graphDef->node(i);

		if (n.name().find("nWeights") != std::string::npos) {
			vNames.push_back(n.name());
			std::cout << n.name()<<std::endl;
		}
	}

	TF_CHECK_OK(m_sess->Run({}, vNames, {}, &out));
	std::cout << "Tensorflow load weight complete" << std::endl;

	////tensorflow::TensorShape inputShape;
	////inputShape.InsertDim(0, 1);
	////inputShape.InsertDim(1, 64);
	////inputShape.InsertDim(2, 64);
	////inputShape.InsertDim(3, 32);
	////inputShape.InsertDim(4, 1);

	// preprocess itk image
	// clip window level
	using WindowFilterType = itk::IntensityWindowingImageFilter<ImageType, ImageType>;
	WindowFilterType::Pointer windowFilter = WindowFilterType::New();
	windowFilter->SetInput(m_inputImage);
	windowFilter->SetOutputMaximum(1000);
	windowFilter->SetWindowMaximum(1000);
	windowFilter->SetOutputMinimum(-300);
	windowFilter->SetWindowMinimum(-300);
	windowFilter->Update();

	// normalize image
	using RescaleFilterType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;
	RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
	rescaleFilter->SetInput(windowFilter->GetOutput());
	rescaleFilter->SetOutputMaximum(255);
	rescaleFilter->SetOutputMinimum(0);
	rescaleFilter->Update();

	// resample image
	using ResampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
	ImageType::SpacingType outputSpacing;
	outputSpacing[0] = 0.2;
	outputSpacing[1] = 0.2;
	outputSpacing[2] = 0.2;

	ImageType::SizeType outputSize;

	ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
	resampleFilter->SetInput(rescaleFilter->GetOutput());
	//resampleFilter->SetInterpolator(2);
	resampleFilter->SetOutputSpacing(outputSpacing);

	std::cout << rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;

	std::cout << "asdfasefasdfas" << std::endl;

	if (rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[0] >= m_patchSize[0] &&
		rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[1] >= m_patchSize[1] &&
		rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[2] >= m_patchSize[2])
	{

		std::cout << "aaaaa" << std::endl;
		for (int i = 0; i < 3; i++)
		{
			outputSize[i] = std::ceil(rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[i] * rescaleFilter->GetOutput()->GetSpacing()[i] / outputSpacing[i]);
		}
	}
	else
	{
		// padding on the image if the input is smaller than network input
		for (int i = 0; i < 3; i++)
		{
			outputSize[i] = m_patchSize[i];
		}
	}
	resampleFilter->SetSize(outputSize);
	resampleFilter->SetOutputOrigin(rescaleFilter->GetOutput()->GetOrigin());
	resampleFilter->SetOutputDirection(rescaleFilter->GetOutput()->GetDirection());
	resampleFilter->Update();

	std::cout << "Image preprocessing complete" << std::endl;

	// prepare image batch indicies
	ImageType::SizeType imageSize = resampleFilter->GetOutput()->GetLargestPossibleRegion().GetSize();

	std::cout << imageSize << std::endl;

	int inum = std::ceil((imageSize[0]-m_patchSize[0])/float(m_stride[0]))+1;
	int jnum = std::ceil((imageSize[1] - m_patchSize[1]) / float(m_stride[1])) + 1;
	int knum = std::ceil((imageSize[2] - m_patchSize[2]) / float(m_stride[2])) + 1;

	std::cout << "ijk num: " << inum << " " << jnum << " " << knum << std::endl;

	//tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1,64,64,32,1 }));
	//
	//auto inputTensorMapped = inputTensor.tensor<float, 5>();

	//float count = 0;

	//for (int i = 0; i < 64; i++)
	//{
	//	for (int j = 0; j < 64; j++)
	//	{
	//		for (int k = 0; k < 32; k++)
	//		{
	//			inputTensorMapped(0, i, j, k, 0) = count;
	//			count++;
	//		}
	//	}
	//}

	//std::vector<std::pair<std::string, tensorflow::Tensor>> input;
	//std::vector<tensorflow::Tensor> answer;

	//input.emplace_back(std::string("images_placeholder:0"), inputTensor);

	//auto statusPred = sess->Run(input, { "predicted_label/prediction:0" }, {}, &answer);

}

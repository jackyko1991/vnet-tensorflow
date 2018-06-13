#include "tf_inference.h"

TF_Inference::TF_Inference()
{
	m_inputImage = ImageType::New();
	m_outputImage = LabelImageType::New();

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

	// create the associate label image
	ImageType::RegionType region = m_outputImage->GetLargestPossibleRegion();
	m_outputImage->SetRegions(region);
	m_outputImage->Allocate();
	m_outputImage->FillBuffer(0);
}

void TF_Inference::SetGraphPath(std::string path)
{
	m_graphPath = path;
}

void TF_Inference::SetCheckpointPath(std::string path)
{
	m_checkpointPath = path;
}

void TF_Inference::SetNumberOfThreads(unsigned int numOfThreads)
{
	// if set to 0, the number of threads used will be maximum number of hardware concurrency, default to be 0
	if (numOfThreads == 0)
	{
	m_numberOfThreads = std::thread::hardware_concurrency();
	}
	else
	{
		m_numberOfThreads = numOfThreads;
	}
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
			//std::cout << n.name()<<std::endl;
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

	if (rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[0] >= m_patchSize[0] &&
		rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[1] >= m_patchSize[1] &&
		rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[2] >= m_patchSize[2])
	{
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

	int patchTotal = 0;
	std::vector <std::shared_ptr<int>> ijkPatchIndicies;

	for (int i = 0; i < inum; i++)
	{
		for (int j = 0; j < jnum; j++)
		{
			for (int k = 0; k < knum; k++)
			{
				//if (patchTotal%m_batchSize == 0)
				//{
					std::shared_ptr<int> ijkPatchIndiciesTmp(new int[6], std::default_delete<int[]>());
					ijkPatchIndicies.push_back(ijkPatchIndiciesTmp);
				//}

				// actually calculate patch indicies
				int istart = i* m_stride[0];
				// for last patch
				if (istart + m_patchSize[0] > imageSize[0])
				{
					istart = imageSize[0] - m_patchSize[0];
				}
				int iend = istart + m_patchSize[0];

				int jstart = j* m_stride[1];
				// for last patch
				if (jstart + m_patchSize[1] > imageSize[1])
				{
					jstart = imageSize[1] - m_patchSize[1];
				}
				int jend = jstart + m_patchSize[1];

				int kstart = k* m_stride[2];
				// for last patch
				if (kstart + m_patchSize[2] > imageSize[2])
				{
					kstart = imageSize[2] - m_patchSize[2];
				}
				int kend = kstart + m_patchSize[2];
				
				ijkPatchIndicies.back().get()[0] = istart;
				ijkPatchIndicies.back().get()[1] = iend;
				ijkPatchIndicies.back().get()[2] = jstart;
				ijkPatchIndicies.back().get()[3] = jend;
				ijkPatchIndicies.back().get()[4] = kstart;
				ijkPatchIndicies.back().get()[5] = kend;

				patchTotal++;
			}
		}
	}

	// create the output label in same size as resampled image
	LabelImageType::Pointer outputImageResampled = LabelImageType::New();
	ImageType::RegionType region = resampleFilter->GetOutput()->GetLargestPossibleRegion();
	outputImageResampled->SetRegions(region);
	outputImageResampled->Allocate();
	outputImageResampled->FillBuffer(0);

	this->BatchInference(resampleFilter->GetOutput(), outputImageResampled, ijkPatchIndicies);
}

ImageType::Pointer CropWithIndicies(ImageType::Pointer input, int* indicies)
{
	std::mutex mutex;
	mutex.lock();
	std::cout << std::this_thread::get_id() << ": " << indicies[0] << " " << indicies[1] << " " << indicies[2] << " " << indicies[3] << " " << indicies[4] << " " << indicies[5] << std::endl;
	mutex.unlock();

	// set indicies to itk region
	ImageType::IndexType start;
	start[0] = indicies[0];
	start[1] = indicies[2];
	start[2] = indicies[4];

	ImageType::SizeType size;
	size[0] = indicies[1] - indicies[0];
	size[1] = indicies[3] - indicies[2];
	size[2] = indicies[5] - indicies[4];

	ImageType::RegionType region(start, size);

	// extract image
	using CropFilter = itk::ExtractImageFilter<ImageType,ImageType>;
	CropFilter::Pointer cropFilter = CropFilter::New();
	cropFilter->SetInput(input);
	cropFilter->SetExtractionRegion(region);
#if ITK_VERSION_MAJOR>=4
	cropFilter->SetDirectionCollapseToIdentity();
#endif
	cropFilter->Update();

	ImageType::Pointer output = ImageType::New();
	output->Graft(cropFilter->GetOutput());

	return output;
}

void TF_Inference::BatchInference(ImageType::Pointer inputImage, LabelImageType::Pointer outputImage, std::vector<std::shared_ptr<int>> patchIndicies)
{
	// create thread pool
	ThreadPool pool(m_numberOfThreads);

	// initialize thread pool
	pool.init();

	//std::map<double*, LabelImageType::Pointer> outputMap;
	std::queue<std::future<ImageType::Pointer>> bufferQueue;
	bool Finish = false;
	int count = 0;
	while (!Finish)
	{
		while (bufferQueue.size() < m_bufferPoolSize)
		{
			std::cout << "Filling up buffer (" << bufferQueue.size() <<"/" << m_bufferPoolSize <<")" <<  std::endl;

			std::cout << "count: " << count <<"/" << patchIndicies.size()<< std::endl;
			std::future<ImageType::Pointer> future = pool.submit(&CropWithIndicies, inputImage, patchIndicies[count].get());
			bufferQueue.push(std::move(future));
			count++;
			if (count == patchIndicies.size())
			{
				std::cout << "break" << std::endl;
				break;
			}
		}

		// convert itk image to tensorflow input
		tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ m_batchSize,m_patchSize[0],m_patchSize[1],m_patchSize[2],1 }));
		auto inputTensorMapped = inputTensor.tensor<float, 5>();

		ImageType::Pointer croppedImage = bufferQueue.front().get();
		bufferQueue.pop();
		itk::ImageRegionIteratorWithIndex<ImageType> imageIterator(croppedImage, croppedImage->GetLargestPossibleRegion());
		
		//croppedImage->Print(std::cout);

		while (!imageIterator.IsAtEnd())
		{
			inputTensorMapped(0,
				imageIterator.GetIndex()[0] - croppedImage->GetLargestPossibleRegion().GetIndex()[0],
				imageIterator.GetIndex()[1] - croppedImage->GetLargestPossibleRegion().GetIndex()[1],
				imageIterator.GetIndex()[2] - croppedImage->GetLargestPossibleRegion().GetIndex()[2],
				0) 
				= imageIterator.Get();
			++imageIterator;
		}

		std::vector<std::pair<std::string, tensorflow::Tensor>> input;
		std::vector<tensorflow::Tensor> predict;

		input.emplace_back(std::string("images_placeholder:0"), inputTensor);
		auto statusPred = m_sess->Run(input, { "predicted_label/prediction:0" }, {}, &predict);
		auto outputTensorMapped = predict[0].tensor<long long int, 4>();

		// convert output tensor to itk label image
		LabelImageType::Pointer croppedLabel = LabelImageType::New();
		croppedLabel->SetRegions(croppedImage->GetLargestPossibleRegion());
		croppedLabel->Allocate();
		itk::ImageRegionIteratorWithIndex<LabelImageType> labelIterator(croppedLabel, croppedLabel->GetLargestPossibleRegion());
		while (!labelIterator.IsAtEnd())
		{
			//std::cout << outputTensorMapped(0, imageIterator.GetIndex()[0], imageIterator.GetIndex()[1], imageIterator.GetIndex()[2]) <<std::endl;
			labelIterator.Set(outputTensorMapped(0, labelIterator.GetIndex()[0], labelIterator.GetIndex()[1], labelIterator.GetIndex()[2]));
			++labelIterator;
		}

		itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
		writer->SetInput(croppedImage);
		writer->SetFileName("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/test/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/patch.nii.gz");
		writer->Write();

		itk::ImageFileWriter<LabelImageType>::Pointer writer2 = itk::ImageFileWriter<LabelImageType>::New();
		writer2->SetInput(croppedLabel);
		writer2->SetFileName("D:/projects/Deep_Learning/tensorflow/vnet-tensorflow/data/raw_data/nii/test/13302970698_20170717_2.16.840.114421.12234.9553621213.9585157213/patch_label.nii.gz");
		writer2->Write();


		system("pause");

		// get data back from thread pool
		//bufferQueue.front().get()->Print(std::cout);
		

		// convert itk image to tensorflow tensor
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


		//system("pause");

		if (count == patchIndicies.size())
			Finish = true;
	}

	pool.shutdown();
}

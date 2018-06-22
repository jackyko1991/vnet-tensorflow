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
	m_outputImage->SetDirection(image->GetDirection());
	m_outputImage->SetOrigin(image->GetOrigin());
	m_outputImage->SetSpacing(image->GetSpacing());
	m_outputImage->Allocate();
	m_outputImage->FillBuffer(0);
}

LabelImageType::Pointer TF_Inference::GetOutput()
{
	return m_outputImage;
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

void TF_Inference::SetBufferPoolSize(unsigned int size)
{
	m_bufferPoolSize = size;
}

void extractIntegerWords(std::string str, std::vector<int>& nums)
{

	std::stringstream ss;

	/* Storing the whole string into string stream */
	ss << str;

	/* Running loop till the end of the stream */
	std::string temp;
	int found;
	while (!ss.eof()) {

		/* extracting word by word from stream */
		ss >> temp;

		/* Checking the given word is integer or not */
		if (std::stringstream(temp) >> found)
		{
			nums.push_back(found);
		}
			
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

		std::string placholderStr = "images_placeholder";
		if (n.name().find(placholderStr) != std::string::npos)
		{
			tensorflow::AttrValue value = n.attr().at("shape");
			
			//extract input shape of network, suppose to be work with 
			//
			//auto shape = graph_def.node().Get(0).attr().at("shape").shape();
			//for (int i = 0; i < shape.dim_size(); i++) {
			//	std::cout << shape.dim(i).size() << std::endl;
			//}
			//
			// but tf c++ api seems not containing tensorshapeproto, use string to extract proper input size instead

			std::vector<int> shape;
			extractIntegerWords(value.DebugString(), shape);

			// set patch size fit input placeholder
			m_patchSize[0] = shape[1];
			m_patchSize[1] = shape[2];
			m_patchSize[2] = shape[3];
		}

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
	windowFilter->SetOutputMinimum(-1000);
	windowFilter->SetWindowMinimum(-1000);
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
	ImageType::SpacingType outputResampledSpacing;
	outputResampledSpacing[0] = 0.2;
	outputResampledSpacing[1] = 0.2;
	outputResampledSpacing[2] = 0.2;

	ImageType::SizeType outputSize;

	using BSplineInterpolatorType = itk::BSplineInterpolateImageFunction<ImageType, double>;
	BSplineInterpolatorType::Pointer bsInterpolator = BSplineInterpolatorType::New();

	ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
	resampleFilter->SetInput(rescaleFilter->GetOutput());
	resampleFilter->SetInterpolator(bsInterpolator);
	resampleFilter->SetOutputSpacing(outputResampledSpacing);

	if (rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[0] >= m_patchSize[0] &&
		rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[1] >= m_patchSize[1] &&
		rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[2] >= m_patchSize[2])
	{
		for (int i = 0; i < 3; i++)
		{
			outputSize[i] = std::ceil(rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[i] * rescaleFilter->GetOutput()->GetSpacing()[i] / outputResampledSpacing[i]);
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
	LabelImageType::Pointer outputLabelResampled = LabelImageType::New();
	ImageType::RegionType region = resampleFilter->GetOutput()->GetLargestPossibleRegion();
	outputLabelResampled->SetRegions(region);
	outputLabelResampled->Allocate();
	outputLabelResampled->FillBuffer(0);
	outputLabelResampled->SetOrigin(resampleFilter->GetOutput()->GetOrigin());
	outputLabelResampled->SetDirection(resampleFilter->GetOutput()->GetDirection());
	outputLabelResampled->SetSpacing(resampleFilter->GetOutput()->GetSpacing());

	this->BatchInference(resampleFilter->GetOutput(), outputLabelResampled, ijkPatchIndicies);

	// reseample the output label back to input space
	using NNInterpolatorType = itk::NearestNeighborInterpolateImageFunction<LabelImageType, double>;
	NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();

	using ResampleLabelFilterType = itk::ResampleImageFilter<LabelImageType, LabelImageType>;
	ResampleLabelFilterType::Pointer resampleLabelFilter = ResampleLabelFilterType::New();
	resampleLabelFilter->SetInput(outputLabelResampled);
	resampleLabelFilter->SetInterpolator(nnInterpolator);
	resampleLabelFilter->SetOutputSpacing(m_inputImage->GetSpacing());
	resampleLabelFilter->SetSize(m_inputImage->GetLargestPossibleRegion().GetSize());
	resampleLabelFilter->SetOutputOrigin(m_inputImage->GetOrigin());
	resampleLabelFilter->SetOutputDirection(m_inputImage->GetDirection());
	resampleLabelFilter->Update();

	m_outputImage->Graft(resampleLabelFilter->GetOutput());
}

ImageType::Pointer CropWithIndicies(ImageType::Pointer input, int* indicies, std::mutex* mutex)
{
	//std::mutex mutex;
	mutex->lock();
	//std::cout << std::this_thread::get_id() << ": " << indicies[0] << " " << indicies[1] << " " << indicies[2] << " " << indicies[3] << " " << indicies[4] << " " << indicies[5] << std::endl;
	
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

	mutex->unlock();

	return output;
}

void TF_Inference::BatchInference(ImageType::Pointer inputImage, LabelImageType::Pointer outputLabel, std::vector<std::shared_ptr<int>> patchIndicies)
{
	// create thread pool
	ThreadPool pool(m_numberOfThreads);

	// initialize thread pool
	pool.init();

	// initialize a mutex
	std::mutex mutex;

	// create a weight label to eliminate overlapping region
	LabelImageType::Pointer weightImage = LabelImageType::New();
	weightImage->SetRegions(outputLabel->GetLargestPossibleRegion());
	weightImage->Allocate();
	weightImage->SetDirection(outputLabel->GetDirection());
	weightImage->SetOrigin(outputLabel->GetOrigin());
	weightImage->SetSpacing(outputLabel->GetSpacing());
	weightImage->FillBuffer(0);

	std::queue<std::future<ImageType::Pointer>> bufferQueue;
	bool Finish = false;
	int count = 0;
	int count2 = 0;
	while (!Finish)
	{
		while (bufferQueue.size() < m_bufferPoolSize && patchIndicies.size()-count2 > m_bufferPoolSize)
		{
			//std::cout << "Filling up buffer (" << bufferQueue.size()+1 <<"/" << m_bufferPoolSize <<")" <<  std::endl;

			//std::cout << "count: " << count+1 <<"/" << patchIndicies.size()<< std::endl;
			std::future<ImageType::Pointer> future = pool.submit(&CropWithIndicies, inputImage, patchIndicies[count].get(), &mutex);
			bufferQueue.push(std::move(future));
			count++;
			if (count == patchIndicies.size())
			{
				break;
			}
		}

		// convert itk image to tensorflow input
		tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ m_batchSize,m_patchSize[0],m_patchSize[1],m_patchSize[2],1 }));
		auto inputTensorMapped = inputTensor.tensor<float, 5>();

		ImageType::Pointer croppedImage = bufferQueue.front().get();
		bufferQueue.pop();
		if (patchIndicies.size() - count2 > m_bufferPoolSize)
		{
			// immediately insert a new job when queue is empty
			std::future<ImageType::Pointer> future = pool.submit(&CropWithIndicies, inputImage, patchIndicies[count].get(), &mutex);
			bufferQueue.push(std::move(future));
			count++;
		}

		itk::ImageRegionIteratorWithIndex<ImageType> imageIterator(croppedImage, croppedImage->GetLargestPossibleRegion());

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

		itk::ImageRegionIteratorWithIndex<LabelImageType> labelIterator(outputLabel, croppedImage->GetLargestPossibleRegion());
		itk::ImageRegionIteratorWithIndex<LabelImageType> weightIterator(weightImage, croppedImage->GetLargestPossibleRegion());

		// iterators need to run separately
		while (!labelIterator.IsAtEnd())
		{
			labelIterator.Set(labelIterator.Get()+
				outputTensorMapped(
					0, 
					labelIterator.GetIndex()[0] - croppedImage->GetLargestPossibleRegion().GetIndex()[0],
					labelIterator.GetIndex()[1] - croppedImage->GetLargestPossibleRegion().GetIndex()[1],
					labelIterator.GetIndex()[2] - croppedImage->GetLargestPossibleRegion().GetIndex()[2]));
			++labelIterator;
		}

		while (!weightIterator.IsAtEnd())
		{
			weightIterator.Set(weightIterator.Get() + 1);
			++weightIterator;
		}

		if (count2 % int(patchIndicies.size()*0.01) == 0)
		{
			std::cout << "Progress: " << count2 + 1 << "/" << patchIndicies.size() << std::endl;
		}
		count2++;

		if (count2 == patchIndicies.size())
			Finish = true;
	}

	pool.shutdown();

	// cast the label to and weight to float image
	itk::CastImageFilter<LabelImageType, ImageType>::Pointer upcaster1 = itk::CastImageFilter<LabelImageType, ImageType>::New();
	upcaster1->SetInput(outputLabel);
	upcaster1->Update();

	itk::CastImageFilter<LabelImageType, ImageType>::Pointer upcaster2 = itk::CastImageFilter<LabelImageType, ImageType>::New();
	upcaster2->SetInput(weightImage);
	upcaster2->Update();

	// divide label by weight
	itk::DivideImageFilter<ImageType, ImageType, ImageType>::Pointer divideFilter = itk::DivideImageFilter<ImageType, ImageType, ImageType>::New();
	divideFilter->SetInput1(upcaster1->GetOutput());
	divideFilter->SetInput2(upcaster2->GetOutput());
	divideFilter->Update();

	// cast the output label back to short, add 0.5 to each pixel to avoid round down issue
	itk::AddImageFilter<ImageType, ImageType>::Pointer addFilter = itk::AddImageFilter<ImageType, ImageType>::New();
	addFilter->SetInput1(divideFilter->GetOutput());
	addFilter->SetConstant2(0.5);
	addFilter->Update();

	itk::CastImageFilter<ImageType, LabelImageType>::Pointer downcaster = itk::CastImageFilter<ImageType, LabelImageType>::New();
	downcaster->SetInput(addFilter->GetOutput());
	downcaster->Update();

	outputLabel->Graft(downcaster->GetOutput());
}

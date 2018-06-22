#ifndef TF_INFERENCE_H
#define TF_INFERENCE_H

#include "itkImage.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkDivideImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkCastImageFilter.h"
#include "itkAddImageFilter.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include <thread>
#include <future>
#include <mutex>
#include "ThreadPool.h"
#include <sstream>
#include <iostream>

typedef itk::Image<float, 3> ImageType;
typedef itk::Image<short, 3> LabelImageType;

class TF_Inference
{
public:
	TF_Inference();
	~TF_Inference();

	void SetImage(ImageType::Pointer);
	LabelImageType::Pointer GetOutput();
	void SetGraphPath(std::string);
	void SetCheckpointPath(std::string);
	void SetNumberOfThreads(unsigned int);
	void SetBufferPoolSize(unsigned int);
	void Inference();

private:
	std::string m_graphPath;
	std::string m_checkpointPath;
	ImageType::Pointer m_inputImage;
	LabelImageType::Pointer m_outputImage;
	tensorflow::Session* m_sess;
	tensorflow::SessionOptions m_options;
	tensorflow::GraphDef* m_graphDef;

	int m_patchSize[3] = { 64,64,32 };
	int m_stride[3] = { 64,64,32 };
	int m_batchSize = 1;

	void BatchInference(ImageType::Pointer, LabelImageType::Pointer, std::vector<std::shared_ptr<int>>);
	//void CropWithIndicies(ImageType::Pointer, ImageType::Pointer, int*);

	// threaded batch inference
	int m_numberOfThreads = std::thread::hardware_concurrency();
	int m_bufferPoolSize = 6;
};

#endif
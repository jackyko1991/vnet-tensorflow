# VNet Tensorflow C++ Inference

C++ implementation of the V-Net architecture for medical image segmentation inferencing

## Dependencies
Known good build dependencies:

- [CMake 3.9.0](https://cmake.org/download/)
	- Following the CMake installation wizard to install CMake binary 

- [ITK 4.13.0](https://github.com/InsightSoftwareConsortium/ITK/tree/v4.13.0)
	- Please compile from source code with following CMake configurations
	
- [Tensorflow 1.8.0 with C++ API](https://github.com/jackyko1991/tensorflow/tree/master/tensorflow/contrib/cmake)
	- Please follow the CMake compilation instruction to build the C++ interface library. i.e. `tensorflow_BUILD_SHARED_LIB` options.

- [Protobuf 3.5.0](https://github.com/google/protobuf/tree/3.5.x)
	- The dependencies will be auto generated via Tensorflow C++ API superbuild.
	- Standalone build from source is possible but not recommended.

Build pass on Windows 10 with MSVC 2015. Test on your own on other platforms and compilers.

## Build from source
1. Specify C++ source folder and target build directory with CMake (GUI/CCMake recommended)
2. Configure and provide necessary dependencies
3. Generate and Build

## Prepare Tensorflow graph from python to C++
In python side we store the checkpoint in metagraph style for simple checkpoint loading. In C++ Tensorflow need to load graph and weight separately and we are now providing the [meta_to_pb.py](../meta_to_pb.py) for the checkpoint conversion.

## Points to note
- Currently the input and output only support absolute path in [main.cxx](./main.cxx)
- We only illustrate one possible data type input with NIFTI format for convenience. It is possible to use JPG, TIFF or other image storage format.
- Only single batch inference is supported in present stage.
- Multi-threaded patch preparation is proposed for GPU utilization. This function is highly experimental and would like to request for development support.
- `m_patchSize` need to be same as the input placeholder size
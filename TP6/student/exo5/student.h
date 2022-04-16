#pragma once

#include <StudentWork.h>
#include <vector>
#include <OPP_cuda.cuh>


class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	void run_filter(
		OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
		OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
		OPP::CUDA::DeviceBuffer<float>& dev_filter,
		const unsigned imageWidth, 
		const unsigned imageHeight,
		const unsigned filterWidth
	);
	
};
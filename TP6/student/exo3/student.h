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

	void run_thumbnail_gather(
		OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
		OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
		OPP::CUDA::DeviceBuffer<uchar2>& dev_map,
		const unsigned imageWidth, 
		const unsigned imageHeight
	);
	
};
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

	void run_thumbnail(
		OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
		OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
		const uchar3 borderColor,
		const unsigned borderSize, 
		const unsigned imageWidth, 
		const unsigned imageHeight
	);
	
};
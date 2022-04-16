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

	void run_RGB2HSV(
		OPP::CUDA::DeviceBuffer<uchar3>& dev_source,
		OPP::CUDA::DeviceBuffer<float>& dev_Hue,
		OPP::CUDA::DeviceBuffer<float>& dev_Saturation,
		OPP::CUDA::DeviceBuffer<float>& dev_Value,
		const unsigned width,
		const unsigned height
	);
	
	void run_HSV2RGB(
		OPP::CUDA::DeviceBuffer<float>& dev_Hue,
		OPP::CUDA::DeviceBuffer<float>& dev_Saturation,
		OPP::CUDA::DeviceBuffer<float>& dev_Value,
		OPP::CUDA::DeviceBuffer<uchar3>& dev_result,
		const unsigned width,
		const unsigned height
	);
	
};
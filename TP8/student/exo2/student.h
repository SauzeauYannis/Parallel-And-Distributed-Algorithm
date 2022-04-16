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

	void run_Histogram(
		OPP::CUDA::DeviceBuffer<float>& dev_value,
		OPP::CUDA::DeviceBuffer<unsigned>& dev_histogram,
		const unsigned width,
		const unsigned height
	);
	
};
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

	void run_binary_map(
		OPP::CUDA::DeviceBuffer<int>& dev_a,
		OPP::CUDA::DeviceBuffer<int>& dev_b,
		OPP::CUDA::DeviceBuffer<int>& dev_result
	);
	
};
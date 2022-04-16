#pragma once

#include <StudentWork.h>
#include <vector>
#include <OPP_cuda.cuh>


class StudentWorkImpl: public StudentWork
{
public:

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	void run_blockEffect(
		OPP::CUDA::DeviceBuffer<float>& dev_source,
		OPP::CUDA::DeviceBuffer<float>& dev_result,
		const unsigned nbWarps
	);

	bool isImplemented() const {
		return true;
	}
	
};
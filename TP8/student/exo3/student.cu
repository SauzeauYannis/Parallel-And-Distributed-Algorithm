#include <iostream>
#include <exo3/student.h>
#include <OPP_cuda.cuh>

namespace 
{
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_Repartition(
	OPP::CUDA::DeviceBuffer<unsigned>& dev_histogram,
	OPP::CUDA::DeviceBuffer<unsigned>& dev_repartition
) {
	OPP::CUDA::inclusiveScan<unsigned, OPP::CUDA::Plus<unsigned>>(dev_histogram, dev_repartition, OPP::CUDA::Plus<unsigned>());
}

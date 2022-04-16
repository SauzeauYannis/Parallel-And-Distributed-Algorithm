#pragma once
#include <OPP_cuda.cuh>

namespace
{
	template<typename T, typename Functor>
	__global__
	void kernelBinaryMap(
			const T* const dev_a, 
			const T* const dev_b, 
			T* const dev_result,
			const Functor& functor,
			const unsigned size
	) {
			const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
			
			if (tid < size)
	 			dev_result[tid] = functor(dev_a[tid], dev_b[tid]);			
	}

	template<typename T, typename Functor>
	void BinaryMap(
		OPP::CUDA::DeviceBuffer<int>& dev_a,
		OPP::CUDA::DeviceBuffer<int>& dev_b,
		OPP::CUDA::DeviceBuffer<int>& dev_result,
		Functor&& functor
	) {
			const unsigned size = dev_a.getNbElements();
			
			const dim3 threads(1024);
			const dim3 blocs((size + 1024 - 1) / 1024);
			
			kernelBinaryMap<<<blocs, threads>>>(
					dev_a.getDevicePointer(),
					dev_b.getDevicePointer(),
					dev_result.getDevicePointer(),
					functor,
					size
			);
	}
}
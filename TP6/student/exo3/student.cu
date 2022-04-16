#include <iostream>
#include <exo3/student.h>
#include <OPP_cuda.cuh>
#include <exo3/mapFunctor.h>

namespace 
{
	template<typename T, typename Functor>
	__global__
	void kernelGather(
			const T* const dev_input, 
			T* const dev_output, 
			Functor map
	) {
			const unsigned tidX = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned tidY = blockIdx.y * blockDim.y + threadIdx.y;
			
			if (tidX < map.imageWidth && tidY < map.imageHeight) {
					const unsigned offset = tidX + tidY * map.imageWidth;

	 				dev_output[offset] = dev_input[map[offset]];
			}		
	}


	template<typename T, typename Functor>
	__host__
	void Gather(
		OPP::CUDA::DeviceBuffer<T>& dev_input,
		OPP::CUDA::DeviceBuffer<T>& dev_output,
		Functor& map
	) {
			const dim3 threads(32, 32);
			const dim3 blocs((map.imageWidth + 32 - 1) / 32, 
	                     (map.imageHeight + 32 - 1) / 32);
			
			kernelGather<<<blocs, threads>>>(
					dev_input.getDevicePointer(),
					dev_output.getDevicePointer(),
					map
			);
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_thumbnail_gather(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
	OPP::CUDA::DeviceBuffer<uchar2>& dev_map,
	const unsigned imageWidth, 
	const unsigned imageHeight
) {
	::MapFunctor<3> map(
		dev_map.getDevicePointer(),
		imageWidth,
		imageHeight
	);

	::Gather<uchar3,MapFunctor<3>>(
		dev_inputImage, dev_outputImage, map
	);
}

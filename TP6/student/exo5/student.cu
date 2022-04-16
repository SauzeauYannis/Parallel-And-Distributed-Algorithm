#include <iostream>
#include <exo5/student.h>
#include <OPP_cuda.cuh>

namespace 
{
	// Vous utiliserez ici les types uchar3 et float3 (internet : CUDA uchar3)
	// Addition de deux "float3"
	__device__ 
	float3 operator+(const float3 &a, const float3 &b)
	{
		return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
		__device__ 
	float3 operator*(const float &a, const uchar3 &b)
	{
		return make_float3(a* b.x, a * b.y, a * b.z);
	}

	__global__
	void kernelFilter(
			const uchar3* const dev_input,
			uchar3* const dev_output,
		  float* filter,
			const unsigned imageWidth,
			const unsigned imageHeight,
			const unsigned filterWidth
	) {
			const unsigned tidX = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned tidY = blockIdx.y * blockDim.y + threadIdx.y;
			 
			if (tidX < imageWidth && tidY < imageHeight) {
					const unsigned tid = tidX + tidY * imageWidth;
					float3 res = make_float3(0.0f, 0.0f, 0.0f);
			 
					for (unsigned i = 0; i < filterWidth; i++) {
							int X = tidX + i - filterWidth / 2;

							if (X < 0) {
									X = -X;
							} else if (X > imageWidth) {
									X = imageWidth - (X - imageWidth);
							}

							for (unsigned j = 0; j < filterWidth; j++) {
									int Y = tidY + j - filterWidth / 2;

									if (Y < 0) {
											Y = -Y;
									} else if (Y > imageHeight) {
											Y = imageHeight - (Y - imageHeight);
									}

									res = res + filter[i * filterWidth + j] * dev_input[X + Y * imageWidth];
							}
					}

					dev_output[tid] = make_uchar3(
							static_cast<unsigned char>(res.x), 
							static_cast<unsigned char>(res.y), 
							static_cast<unsigned char>(res.z));			
			}
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_filter(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
	OPP::CUDA::DeviceBuffer<float>& dev_filter,
	const unsigned imageWidth, 
	const unsigned imageHeight,
	const unsigned filterWidth
) {
		const unsigned size = dev_inputImage.getNbElements();
			 
		const dim3 threads(32, 32);
	  const dim3 blocs((imageWidth + 32 - 1) / 32, 
	                   (imageHeight + 32 - 1) / 32);
			
		kernelFilter<<<blocs, threads>>>(
				dev_inputImage.getDevicePointer(),
				dev_outputImage.getDevicePointer(),
				dev_filter.getDevicePointer(),
				imageWidth,
				imageHeight,
				filterWidth
			);
}

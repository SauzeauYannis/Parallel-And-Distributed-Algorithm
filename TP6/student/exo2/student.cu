#include <iostream>
#include <exo2/student.h>
#include <OPP_cuda.cuh>

namespace 
{
	// NB : la fonction ci-dessous sert principalement à rendre le code plus lisible.
	//  Selon ce principe, plus une fonction est courte, et plus il est facile de la comprendre,
	//  et par effet de bord de la maintenir et déverminer ...
	template<int TSIZE=3>
	__device__
	bool isOnBorder(
		const unsigned x,
		const unsigned y,
		const unsigned borderSize, 
		const unsigned imageWidth, 
		const unsigned imageHeight
	) {
		const auto thumbnailWidth = imageWidth / TSIZE;
		const auto xInBlock = x % thumbnailWidth;
		const auto thumbnailHeight = imageHeight / TSIZE;
		const auto yInBlock = y % thumbnailHeight;
		return 
			xInBlock < borderSize || 
			yInBlock < borderSize || 
			xInBlock >= (thumbnailWidth-borderSize) || 
			yInBlock >= (thumbnailHeight-borderSize);
	}

	template<int TSIZE=3>
	__global__
	void thumbnailKernel(
		const uchar3 * const input, 
		uchar3 * const output, 
		const uchar3 borderColor, 
		const unsigned borderSize, 
		const unsigned imageWidth, 
		const unsigned imageHeight
	) {
			const unsigned tidX = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned tidY = blockIdx.y * blockDim.y + threadIdx.y;
			
			if (tidX < imageWidth && tidY < imageHeight) {
					const unsigned offset = tidX + tidY * imageWidth;

					output[offset] = 
							isOnBorder<TSIZE>(tidX, tidY, borderSize, imageWidth, imageHeight) ?
							borderColor : input[offset];
			}
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_thumbnail(
	OPP::CUDA::DeviceBuffer<uchar3>& dev_inputImage,
	OPP::CUDA::DeviceBuffer<uchar3>& dev_outputImage,
	const uchar3 borderColor,
	const unsigned borderSize,
	const unsigned imageWidth, 
	const unsigned imageHeight
) {
	const dim3 threads(32, 32);
	const dim3 blocs((imageWidth + 32 - 1) / 32, 
	                 (imageHeight + 32 - 1) / 32);

	thumbnailKernel<<<blocs, threads>>>(
			dev_inputImage.getDevicePointer(),
			dev_outputImage.getDevicePointer(),
			borderColor,
			borderSize,
			imageWidth,
			imageHeight
	);
}
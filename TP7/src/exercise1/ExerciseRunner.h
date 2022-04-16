#pragma once
#include <ppm.h>
#include <exo1/student.h>

namespace
{
    template<int channel>
    __global__
    void separateChannelKernel(
        uchar3 const*const input, 
        float *const result, 
        const size_t width, 
        const size_t height
    ) {
        const auto x = threadIdx.x + blockIdx.x * blockDim.x;
        const auto y = threadIdx.y + blockIdx.y * blockDim.y;
        
        const auto tidInBlock = threadIdx.x + blockDim.x * threadIdx.y;
        const auto blockId    =  blockIdx.x + gridDim.x  *  blockIdx.y;

        const auto tidOutput = tidInBlock + 1024 * blockId;
        if( x < width && y < height ) {
            const auto tidInput = x + width * y;
            uchar const*const input_linear = reinterpret_cast<uchar const*>(input);
            result[tidOutput] = input_linear[channel + 3*tidInput];
        }
        else
            result[tidOutput] = (uchar)0;
    }

    __global__
    void regroupChannelKernel(
        float const*const inputRed, 
        float const*const inputGreen,
        float const*const inputBlue, 
        uchar3 *const result, 
        const size_t width,
        const size_t height
    ) {
        const auto x = threadIdx.x + blockIdx.x * blockDim.x;
        if( x >= width ) return;
        
        const auto y = threadIdx.y + blockIdx.y * blockDim.y;
        if( y >= height ) return;
        
        const auto tidInBlock = threadIdx.x + blockDim.x * threadIdx.y;
        const auto blockId    =  blockIdx.x +  gridDim.x *  blockIdx.y;

        const auto divider = float(
            fminf(width - blockIdx.x * blockDim.x, 32) * fminf(height - blockIdx.y * blockDim.y, 32)
        );

        const auto tidInput = tidInBlock + blockDim.x * blockDim.y * blockId;
        
        const auto tidOutput = x + width * y;
        result[tidOutput].x = static_cast<uchar>(inputRed[tidInput] / divider);
        result[tidOutput].y = static_cast<uchar>(inputGreen[tidInput] / divider);
        result[tidOutput].z = static_cast<uchar>(inputBlue[tidInput] / divider);
    }
};

class ExerciseRunner 
{
    using uchar = unsigned char;
    
    const dim3 threads;
    const unsigned width;
    const unsigned height;
    const dim3 blocks;
    const unsigned size;
    const unsigned sizeForAll;

    OPP::CUDA::DeviceBuffer<float> dev_inputR;
    OPP::CUDA::DeviceBuffer<float> dev_inputG;
    OPP::CUDA::DeviceBuffer<float> dev_inputB;
    OPP::CUDA::DeviceBuffer<float> dev_outputR;
    OPP::CUDA::DeviceBuffer<float> dev_outputG;
    OPP::CUDA::DeviceBuffer<float> dev_outputB;

public:
    ExerciseRunner(PPMBitmap const*const input) :
        threads(32,32),
        width(input->getWidth()), height(input->getHeight()), 
        blocks((width+31)/32, (height+31)/32),
        size(input->getWidth() * input->getHeight()),
        sizeForAll(1024 * blocks.x * blocks.y),
        dev_inputR(sizeForAll),
        dev_inputG(sizeForAll),
        dev_inputB(sizeForAll),
        dev_outputR(sizeForAll),
        dev_outputG(sizeForAll),
        dev_outputB(sizeForAll)
    {
        OPP::CUDA::DeviceBuffer<uchar3> dev_inputRGB(
            reinterpret_cast<uchar3*>(input->getPtr()), size);
        ::separateChannelKernel<0><<<blocks, threads>>>(
            dev_inputRGB.getDevicePointer(), dev_inputR.getDevicePointer(), width, height);
        ::separateChannelKernel<1><<<blocks, threads>>>(
            dev_inputRGB.getDevicePointer(), dev_inputG.getDevicePointer(), width, height);
        ::separateChannelKernel<2><<<blocks, threads>>>(
            dev_inputRGB.getDevicePointer(), dev_inputB.getDevicePointer(), width, height);    
    }

    void run(StudentWorkImpl*impl) 
    {
        impl->run_blockEffect(dev_inputR, dev_outputR);
        impl->run_blockEffect(dev_inputG, dev_outputG);
        impl->run_blockEffect(dev_inputB, dev_outputB);
    }

    void copyTo(PPMBitmap* destImage)
    {
	    OPP::CUDA::DeviceBuffer<uchar3> dev_outputRGB(size);
        ::regroupChannelKernel<<<blocks, threads>>>(
            dev_outputR.getDevicePointer(), 
            dev_outputG.getDevicePointer(), 
            dev_outputB.getDevicePointer(), 
            dev_outputRGB.getDevicePointer(), 
            width, height
        );
        dev_outputRGB.copyToHost(
            reinterpret_cast<uchar3*>(destImage->getPtr()));
    }
    
};


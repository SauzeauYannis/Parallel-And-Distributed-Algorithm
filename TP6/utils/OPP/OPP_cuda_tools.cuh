#pragma once

#include <helper_cuda.h>

namespace OPP 
{
    namespace CUDA 
    {
        template<typename T>
        __device__ 
        inline T* getSharedMemory() 
        {
            // declare the global extern pointer
            extern __shared__ char externGlobalPtr[];
            // return it after cast to T*
            return reinterpret_cast<T*>(externGlobalPtr);
        }


        template<typename T>
        T* allocateDeviceArray(const unsigned size) 
        {
            const auto sizeInBytes = sizeof(T) * size;
            T*result;
            cudaMalloc(&result, sizeInBytes);
            return result;
        }

        template<typename T>
        __host__
        T* allocateDeviceArrayAndCopyFromHost(const unsigned size, T const*const h_data) 
        {
            const auto sizeInBytes = sizeof(T) * size;
            T*result;
            cudaMalloc(&result, sizeInBytes);
            cudaMemcpy(result, h_data, sizeInBytes, cudaMemcpyHostToDevice);
            return result;
        }

        template<typename T>
        __host__
        T getSMPNumber() 
        {
            int device;
            cudaGetDevice(&device);
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            return static_cast<T>(props.multiProcessorCount);
        }
    }
}
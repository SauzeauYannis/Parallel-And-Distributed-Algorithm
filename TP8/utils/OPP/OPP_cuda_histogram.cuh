#pragma once

#include <OPP/OPP_cuda_tools.cuh>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace OPP
{
    namespace CUDA 
    {
        namespace HISTOGRAM
        {
            template<typename T, typename U, typename FunctorToBin>
            __global__
            void unsharedHistogramKernel(
                T const*const input,
                const unsigned size, 
                U *const histo, 
                const FunctorToBin functor
            ) {
                const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
                if( tid < size )
                {
                    const T& data = input[tid];
                    atomicAdd( &histo[functor(data)], 1 );
                }
            }
            
            template<typename U>
            __device__
            void initializeSharedHistogram()
            {
                U *const sharedHistogram = getSharedMemory<U>();
                sharedHistogram[threadIdx.x] = U(0);
                __syncthreads();
            }

            template<typename T, typename U, typename FunctorToBin>
            __device__
            void computeHistogramPerBlock(
                T const*const input,
                const unsigned size, 
                const FunctorToBin functor
            ) {
                U *const sharedHistogram = getSharedMemory<U>();
                const auto numberOfChunks = gridDim.x;
                const auto chunkSize = ( size + numberOfChunks -1 ) / numberOfChunks;
                const auto chunkStart = chunkSize * blockIdx.x;
                for( auto tid = threadIdx.x; tid < chunkSize; tid += blockDim.x )
                {
                    const auto indexInInput = chunkStart + tid;
                    if( indexInInput >= size ) break;
                    const T& data = input[indexInInput];
                    atomicAdd( &sharedHistogram[functor(data)], U(1) );
                }
                __syncthreads();
            }

            template<typename T, typename U>
            __device__
            void addPerBlockHistogramToGlobal(U *const histo)
            {
                U const*const sharedHistogram = getSharedMemory<U>(); 
                atomicAdd( &histo[threadIdx.x] , sharedHistogram[threadIdx.x] );                
            }

            template<typename T, typename U, typename FunctorToBin>
            __global__
            void histogramKernel(
                T const*const input,
                const unsigned size, 
                U *const histo, 
                const FunctorToBin functor
            ) {
                initializeSharedHistogram<U>();
                computeHistogramPerBlock<T,U,FunctorToBin>(input, size, functor);
                addPerBlockHistogramToGlobal<T,U>(histo);
            }
        }

        template<typename T, typename U, typename FunctorToBin>
        __host__
        void unsharedHistogram(DeviceBuffer<T>& buffer, DeviceBuffer<U>& histo, const FunctorToBin functor) 
        {
            dim3 threads(256);
            dim3 blocks((buffer.getNbElements() + threads.x - 1) / threads.x);
            cudaMemset(histo.getDevicePointer(), 0, histo.getSizeInBytes());
            HISTOGRAM::unsharedHistogramKernel<T, FunctorToBin> <<<blocks, threads>>>
            (
                buffer.getDevicePointer(), 
                buffer.getNbElements(), 
                histo.getDevicePointer(), 
                functor 
            );
            getLastCudaError("unsharedHistogram returns error");
        }


        template<typename T, typename U, typename FunctorToBin>
        __host__
        void computeHistogram(DeviceBuffer<T>& buffer, DeviceBuffer<U>& histo, const FunctorToBin functor) 
        {
            dim3 threads(256);
            const unsigned nbSMP = getSMPNumber<unsigned>();
            dim3 blocks(std::min(nbSMP * 4u, (buffer.getNbElements() + threads.x - 1) / threads.x));
            cudaMemset(histo.getDevicePointer(), 0, histo.getSizeInBytes());
            HISTOGRAM::histogramKernel<T, U, FunctorToBin> <<<blocks, threads, threads.x * sizeof(U)>>>
            (
                buffer.getDevicePointer(), 
                buffer.getNbElements(), 
                histo.getDevicePointer(), 
                functor 
            );
            getLastCudaError("computeHistogram returns error");
        }

    } // namespace CUDA

} // namespace OPP
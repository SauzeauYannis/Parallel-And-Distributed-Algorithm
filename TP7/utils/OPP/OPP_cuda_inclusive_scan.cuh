#pragma once 

#include <helper_cuda.h>
#include <OPP_cuda.cuh>
#include <exception>

namespace OPP 
{

    namespace CUDA 
    {

        template<typename T, typename Functor>
        void inclusiveScan(DeviceBuffer<T>& data, DeviceBuffer<T>& result, const Functor functor);
        
        namespace SCAN 
        {
            
            template<typename T>
            __device__ inline
            void loadSharedMemory(T const*const data,const unsigned size) 
            {
                T*const shared = getSharedMemory<T>();
                auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
                if( tid < size )
                    shared[threadIdx.x] = data[tid]; 
                __syncthreads();
            }

            template<typename T>
            __device__ inline
            void saveSharedMemory(T *const data, const unsigned size) 
            {
                T*const shared = getSharedMemory<T>();
                auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
                if( tid < size )
                    data[tid] = shared[threadIdx.x]; 
                __syncthreads();
            }

            
            template<typename T, typename Functor>
            __device__ inline
            void scanJumpingStep(Functor functor, const unsigned limitInBlock, const int jump)
            {
                T*const shared = getSharedMemory<T>();
                const auto tid = threadIdx.x;
                T prevValue = shared[tid];
                __syncthreads();
                if(tid+jump < limitInBlock) 
                    shared[tid+jump] = functor(prevValue, shared[tid+jump]); 
                __syncthreads();
            }


            template<typename T, typename Functor>
            __global__
            __launch_bounds__(256,4)
            void iScanPerBlock(T const*const data, const unsigned size, T *const output, T*const offset, Functor functor)
            {
                T*const shared = getSharedMemory<T>();
                loadSharedMemory<T>(data, size);
                const unsigned limitInBlock = umin(blockDim.x, size-blockIdx.x*blockDim.x);
                for(unsigned i=1; i<blockDim.x; i<<=1)
                    scanJumpingStep<T,Functor>(functor, limitInBlock, i);
                saveSharedMemory<T>(output, size);
                if (threadIdx.x == 0)
                    offset[blockIdx.x] = shared[blockDim.x-1];
            }

            
            template<typename T, typename Functor>
            __global__
            void addOffset(T *const data, const unsigned size, T*const offset, Functor functor)
            {
                // take the offset, add it to all values. No need to use shared memory
                const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
                if( tid < size ) 
                    data[tid] = functor(offset[blockIdx.x], data[tid]);
            }

            template<typename T, typename Functor>
            void stepOne(DeviceBuffer<T>& data, DeviceBuffer<T>& result, DeviceBuffer<T>& perBlockOffset, const Functor functor)
            {
                dim3 threads(256);
                dim3 blocks(perBlockOffset.getNbElements());
                iScanPerBlock<T,Functor><<<blocks,threads,sizeof(T)*threads.x>>>(
                    data.getDevicePointer(), 
                    data.getNbElements(), 
                    result.getDevicePointer(), 
                    perBlockOffset.getDevicePointer(), 
                    functor
                );
            }

            template<typename T, typename Functor>
            void stepTwo(DeviceBuffer<T>& data, DeviceBuffer<T>& result, DeviceBuffer<T>& perBlockOffset, const Functor functor)
            {
                // final offset for blocks 1, 2, ... skip the first one!
                if( perBlockOffset.getNbElements() > 1 )
                {
                    inclusiveScan<T,Functor>(perBlockOffset, perBlockOffset, functor);
                    dim3 threads(256);
                    dim3 blocks(perBlockOffset.getNbElements() - 1 );
                    addOffset<T, Functor><<<blocks,threads>>>(
                        result.getDevicePointer()+threads.x, 
                        result.getNbElements()-threads.x,
                        perBlockOffset.getDevicePointer(), 
                        functor
                    );
                }
            }
    
        }

        template<typename T, typename Functor>
        void inclusiveScan(DeviceBuffer<T>& data, DeviceBuffer<T>& result, const Functor functor)
        {
            if( data.getNbElements() != result.getNbElements() )
                throw new std::runtime_error("inclusiveScan call error: bad size");
            const unsigned nbBlocks((data.getNbElements() + 255) / 256);
            DeviceBuffer<T> perBlockOffset(nbBlocks);
            SCAN::stepOne<T,Functor>(data, result, perBlockOffset, functor);
            SCAN::stepTwo<T,Functor>(data, result, perBlockOffset, functor);
        }
        
    } // namespace CUDA

} // namespace OPP
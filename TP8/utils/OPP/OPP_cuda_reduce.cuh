#pragma once

#include <OPP_cuda.cuh>

namespace OPP 
{
    // reduction : opération associative mais non commutative
    // -> aucune permutation autorisé !
    // il faut donc respecter l'ordre des opérations ...
    // NB : possible d'écrire une version "commutative" plus performante
    namespace CUDA 
    {
        namespace REDUCE 
        {
            template<typename T>
            __device__ inline
            void loadSharedMemory(T const*const data,const unsigned size, const T nil) 
            {
                T*const shared = getSharedMemory<T>();
                auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
                shared[threadIdx.x] = tid < size ? data[tid] : nil; 
                __syncthreads();
            }


            template<typename T, typename Functor>
            __device__ inline
            void reduceJumpingStep(Functor functor, const int jump)
            {
                T*const shared = getSharedMemory<T>();
                const auto tid = threadIdx.x;
                if((tid % (jump<<1)) == 0) 
                    shared[tid] = functor(shared[tid], shared[tid+jump]); 
                __syncthreads();
            }


            template<typename T, typename Functor>
            __global__ 
            __launch_bounds__(256,4)
            void reduce_kernel(T const*const data, const unsigned size, T*const resultPerBlock, const Functor functor, const T nil)
            {
                // load data, sync, do reduce, save partial reduce
                T*const shared = getSharedMemory<T>();
                loadSharedMemory<T>(data, size, nil);
                for(int i=1; i<blockDim.x; i<<=1)
                    reduceJumpingStep<T,Functor>(functor, i);
                if( threadIdx.x == 0 )
                    resultPerBlock[blockIdx.x] = shared[0];
            }

        } // namespace REDUCE

        template<typename T, typename Functor>
        __host__ 
        T reduce(DeviceBuffer<T>& data, const Functor functor, const T nil=T(0))
        {
            if( data.getNbElements() == 1 ) {
                T result;
                data.copyToHost(&result);
                return result;    
            }
            dim3 threads(256); // 1<<8
            dim3 blocs((data.getNbElements()+threads.x-1) / threads.x);
            DeviceBuffer<T> resultPerBloc(blocs.x);
            REDUCE::reduce_kernel<<<blocs, threads, sizeof(T)*threads.x>>>(
                data.getDevicePointer(), 
                data.getNbElements(), 
                resultPerBloc.getDevicePointer(), 
                functor, 
                nil
            );
            return reduce<T,Functor>(resultPerBloc, functor, nil);
        }

    } // namespace CUDA

} // namespace OPP
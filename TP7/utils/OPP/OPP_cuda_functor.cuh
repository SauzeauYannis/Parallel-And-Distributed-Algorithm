#pragma once

namespace OPP
{
    namespace CUDA 
    {
        template<typename T>
        struct Plus 
        {
            __device__ __host__
            T operator()(const T&a, const T&b) const { return a+b; }
        };

        template<typename T>
        struct Multiply 
        {
            __device__ __host__ 
            T operator()(const T&a, const T&b) const { return a*b; }
        };
    } // namespace CUDA

} // namespace OPP
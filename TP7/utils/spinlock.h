#pragma once

class SpinLock
{
    int *dev_mutex;
public:
    __host__
    SpinLock() {
        cudaMalloc(&dev_mutex, sizeof(int));
        cudaMemset(dev_mutex, 0, sizeof(int));
    }
    __host__
    ~SpinLock() {
        cudaFree(dev_mutex);
    }
    __device__ 
    void lock() {
        while(atomicCAS(dev_mutex, 0, 1)) ;
    }
    __device__
    void unlock() {
        atomicExch(dev_mutex, 0);
    }
};
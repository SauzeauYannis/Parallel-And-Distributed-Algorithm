#pragma once
#include <helper_cuda.h>

namespace OPP
{
    namespace CUDA 
    {
        template<typename T>
        class DeviceBuffer 
        {
            T * buffer;
            const unsigned nbElements;
            const size_t sizeInBytes;
        public:
            DeviceBuffer(const unsigned nb) : nbElements(nb), sizeInBytes(sizeof(T)*nb) 
            {
                cudaMalloc(&buffer, sizeInBytes);
                getLastCudaError("Unable to allocate a DeviceBuffer ...");
            }
            DeviceBuffer(T const*const host_buffer, const unsigned nb) : nbElements(nb), sizeInBytes(sizeof(T)*nb) 
            {
                cudaMalloc(&buffer, sizeInBytes);
                getLastCudaError("Unable to allocate a DeviceBuffer ...");
                cudaMemcpy(buffer, host_buffer, sizeInBytes, cudaMemcpyHostToDevice);
                getLastCudaError("Unable to initialize a DeviceVector ...");
            }
            DeviceBuffer(const DeviceBuffer& db) : nbElements(db.nbElements), sizeInBytes(db.sizeInBytes) 
            {
                cudaMalloc(&buffer, sizeInBytes);
                getLastCudaError("Unable to allocate a DeviceBuffer ...");
                cudaMemcpy(buffer, db.buffer, sizeInBytes, cudaMemcpyDeviceToDevice);
                getLastCudaError("Unable to initialize a DeviceBuffer ...");
            }
            ~DeviceBuffer() 
            {
                cudaFree(buffer);
                getLastCudaError("Unable to free a DeviceVector ...");
            }
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;
            size_t getSizeInBytes() const 
            { 
                return sizeInBytes; 
            }
            unsigned getNbElements() const 
            { 
                return nbElements; 
            }
            T const * getDevicePointer() const { 
                return buffer; 
            }
            T* getDevicePointer() { 
                return buffer; 
            }
            void copyToHost(T*const host_buffer) const
            {
                cudaMemcpy(host_buffer, buffer, sizeInBytes, cudaMemcpyDeviceToHost);
                getLastCudaError("Unable to copy a DeviceBuffer to a host one ...");
            }
            void copyFromHost(T const*const host_buffer) const
            {
                cudaMemcpy(buffer, host_buffer, sizeInBytes, cudaMemcpyHostToDevice);
                getLastCudaError("Unable to copy a DeviceBuffer from a host one ...");
            }
        };

    } // namespace CUDA

} // namespace OPP
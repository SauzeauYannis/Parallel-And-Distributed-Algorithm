// MPI distributed vector (by block)
#pragma once

#include <mpi.h>
#include <iostream>
#include <stdexcept>

class DistributedBlockVector
{
    struct s_block_info
    {
        int m_rank;   // of the pus
        int m_offset; // local offset for current pus
        int m_size;   // local size (N/p)
    };
    const s_block_info m_block;

    float *m_ptr; // local memory
    MPI_File m_file;

    // obtain the const block of information
    static s_block_info get_block_info(const int N)
    {
        s_block_info blk;
        int p;
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        MPI_Comm_rank(MPI_COMM_WORLD, &blk.m_rank);

        blk.m_size = (N + p - 1) / p;
        blk.m_offset = blk.m_size * blk.m_rank;
        if (N < blk.m_offset + blk.m_size)
            blk.m_size = N - blk.m_offset;

        std::cout <<"["<<blk.m_rank<<","<<p<<"] "<<blk.m_size<<"/"<<blk.m_offset<<std::endl;

        return blk;
    }

  public:

    const int m_N; // size of the distributed vector

    // constructor ... no default constructor
    DistributedBlockVector(const int N, const char *fileName) : m_block(get_block_info(N)), m_N(N)
    {
        MPI_Alloc_mem(
            m_block.m_size * sizeof(float), 
            MPI_INFO_NULL, 
            &m_ptr
        );
        MPI_File_open(
            MPI_COMM_WORLD, fileName, 
            MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, 
            &m_file
        );
        MPI_File_seek(
            m_file, 
            sizeof(float) * m_block.m_offset, 
            MPI_SEEK_SET
        );
        MPI_File_read(
            m_file, 
            m_ptr, 
            m_block.m_size, 
            MPI_FLOAT, 
            MPI_STATUS_IGNORE
        );
    }

    // default constructor is forbidden
    DistributedBlockVector() = delete;
    // idem for copy ...
    DistributedBlockVector(DistributedBlockVector &) = delete;
    DistributedBlockVector operator=(DistributedBlockVector &) = delete;

    // destructor
    ~DistributedBlockVector()
    {
        Synchronize();
        MPI_Free_mem(m_ptr);
        MPI_File_close(&m_file);
    }

    // force write to 
    void Synchronize()
    {
        MPI_File_seek(
            m_file, 
            sizeof(float) * m_block.m_offset, 
            MPI_SEEK_SET
        );
        MPI_File_write(
            m_file, 
            m_ptr, 
            m_block.m_size, 
            MPI_FLOAT, 
            MPI_STATUS_IGNORE
        );
        MPI_File_sync( m_file );
    }

    // return the first value index
    int Start() const { return m_block.m_offset; }

    // return the first bad index after local values
    int End() const { return m_block.m_offset + m_block.m_size; }

    // access to elements ...
    float& operator[](const int i)
    {
        const int idx = i - m_block.m_offset;
        if (idx < 0 || idx >= m_block.m_size)
            throw std::range_error("DistributedBlockVector: Bad access!");
        return m_ptr[idx];
    }

    // idem, for const vector
    float operator[](const int i) const
    {
        const int idx = i - m_block.m_offset;
        if (idx < 0 || idx >= m_block.m_size)
            throw std::range_error("DistributedBlockVector: Bad access!");
        return m_ptr[idx];
    }

    // get the size of the managed block
    //int getSize() const { return m_block.m_size; };
};

// MPI distributed matrix by (block of successive) rows
#pragma once

#include <iostream>
#include <MPI/OPP_MPI.h>

struct MatrixException {
    std::string m_msg;
    MatrixException(const char*const msg) : m_msg(msg) {}
    MatrixException(const MatrixException& me) : m_msg(me.m_msg) {}
};


class DistributedRowMatrix
{
    const OPP::MPI::Communicator communicator;

    struct s_block_info {
        int m_first_row; // begining of the block
        int m_number_of_rows; // size of the block
        int m_offset; // real starting position (real begining)
    };

    s_block_info m_block; // data associated to the block manipulated by this processus

    float* m_ptr; // local copy of the block of data
    MPI_File m_file; // MPI file that contains the data

    // hidden fonctions
    DistributedRowMatrix() = delete;
    DistributedRowMatrix(DistributedRowMatrix&) = delete;
    DistributedRowMatrix operator=(DistributedRowMatrix&) = delete;

    // calculates information about the block managed by the current processus
    static s_block_info get_block_info(const OPP::MPI::Communicator&communicator, const int m, const int n)
    {
        s_block_info blk;
        blk.m_number_of_rows = (m + communicator.size-1) / communicator.size;
        blk.m_first_row = blk.m_number_of_rows * communicator.rank;
        if (blk.m_first_row + blk.m_number_of_rows > m)
            blk.m_number_of_rows = m - blk.m_first_row;
        blk.m_offset = blk.m_first_row * n;
        //std::cout <<"["<<blk.m_rank<<","<<blk.m_world_size<<"] "<<blk.m_number_of_rows<<"/"<<blk.m_first_row<<"/"<<blk.m_offset<<std::endl;
        return blk;
    }

public:

    const int m_m; // number of rows
    const int m_n; // number of columns

    DistributedRowMatrix(const OPP::MPI::Communicator&communicator, const int m, const int n, const char*fileName)
    : communicator(communicator), m_block(get_block_info(communicator, m, n)), m_m(m), m_n(n)
    {
        MPI_Alloc_mem(
            m_n*m_block.m_number_of_rows*sizeof(float), 
            MPI_INFO_NULL, 
            &m_ptr
        );
        MPI_File_open(
            MPI_COMM_WORLD, fileName, 
            MPI_MODE_CREATE | MPI_MODE_RDWR,
            MPI_INFO_NULL, &m_file
        );
        MPI_File_seek(
            m_file, sizeof(float)*m_block.m_offset, MPI_SEEK_SET
        );
        MPI_File_read(
            m_file, 
            m_ptr, 
            m_n*m_block.m_number_of_rows, 
            MPI_FLOAT, 
            MPI_STATUS_IGNORE
        );
    }

    ~DistributedRowMatrix()
    {
        Synchronize();
        MPI_Free_mem(m_ptr);
        MPI_File_close(&m_file);
    }

    // force write to the "distributed" file
    void Synchronize()
    {
        MPI_File_seek( 
            m_file, 
            sizeof(float)*m_block.m_offset, 
            MPI_SEEK_SET
        );
        MPI_File_write(
            m_file, 
            m_ptr, 
            m_n*m_block.m_number_of_rows, 
            MPI_FLOAT, 
            MPI_STATUS_IGNORE
        );
        MPI_File_sync(m_file);
    }

    // return the first row index
    int Start() const { return m_block.m_first_row; }

    // return the first bad row after local values
    int End() const { return m_block.m_first_row+m_block.m_number_of_rows; }

    // access to a row of a DistributedRowMatrix
    class MatrixRow {
        const unsigned m_width;
        float*m_v;
        MatrixRow();
    public:
        MatrixRow(const unsigned row, DistributedRowMatrix&m) :
        m_width(m.m_n), m_v(m.m_ptr+(row-m.m_block.m_first_row)*m.m_n)
        {}
        float&operator[](const int col) {
            if( col<0 || col>=int(m_width) )
                throw MatrixException( "Matrix: column out-of-bound access" );
            return m_v[col];
        }
    };

    MatrixRow operator[](const int row) {
        if( row<Start() || row>=End() )
        throw MatrixException( "Matrix: row out-of-bound access" );
        return MatrixRow(row,*this);
    }

    class ConstMatrixRow {
        const unsigned m_width;
        const float*const m_v;
        ConstMatrixRow();
    public:
        ConstMatrixRow(const unsigned row, const DistributedRowMatrix&m) :
        m_width(m.m_n), m_v(m.m_ptr+(row-m.m_block.m_first_row)*m.m_n)
        {}
        float operator[](const int col) const {
            if( col<0 || col>=int(m_width) )
                throw MatrixException( "Matrix: column out-of-bound access" );
            return m_v[col];
        }
    };

    ConstMatrixRow operator[](const int row) const {
        if( row<Start() || row>=End() )
        throw MatrixException( "Matrix: row out-of-bound access" );
        return ConstMatrixRow(row,*this);
    }

    OPP::MPI::Communicator getCommunicator() const { return communicator; }
};

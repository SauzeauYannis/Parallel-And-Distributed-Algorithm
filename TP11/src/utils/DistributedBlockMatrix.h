// MPI distributed matrix by (block of successive) rows
#pragma once

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstring>

struct MatrixException {
    std::string m_msg;
    MatrixException(const char*const msg) : m_msg(msg) {}
    MatrixException(const MatrixException& me) : m_msg(me.m_msg) {}
};


class DistributedBlockMatrix
{
    struct s_block_info {
        int m_world_width; // number of processes in the MPI world
        int m_world_height; // number of processes in the MPI world
        int m_row_rank; // rank of the current processes
        int m_col_rank; // rank of the current processes
        int m_first_row; // begining of the block
        int m_number_of_rows; // width of the block
        int m_first_column;
        int m_number_of_columns;
        int m_offset; // real starting position (real begining)
    };

    s_block_info m_block; // data associated to the block manipulated by this processus

    float* m_ptr; // local copy of the block of data
    MPI_File m_file; // MPI file that contains the data

    // calculates information about the block managed by the current processus
    static s_block_info get_block_info(const int m, const int n)
    {
        s_block_info blk;
        int world_size, world_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        blk.m_world_height = blk.m_world_width = int(round(sqrt(world_size)));
        blk.m_number_of_rows = (m + blk.m_world_height-1) / blk.m_world_height;
        blk.m_number_of_columns = (n + blk.m_world_width-1) / blk.m_world_width;
        blk.m_row_rank = world_rank / blk.m_world_width;
        blk.m_col_rank = world_rank % blk.m_world_width;
        blk.m_first_row = blk.m_number_of_rows * blk.m_row_rank;
        if (blk.m_first_row + blk.m_number_of_rows > m)
            blk.m_number_of_rows = m - blk.m_first_row;
        blk.m_first_column = blk.m_number_of_columns * blk.m_col_rank;
        if (blk.m_first_column + blk.m_number_of_columns > n)
            blk.m_number_of_columns = n - blk.m_first_column;
        blk.m_offset = blk.m_first_row * n + blk.m_first_column;
        /*std::cout <<"["<<world_rank<<","<<world_size<<"] "
            <<blk.m_number_of_rows<<"/"<<blk.m_first_row<<"/"<<blk.m_row_rank<<" "
            <<blk.m_number_of_columns<<"/"<<blk.m_first_column<<"/"<<blk.m_col_rank<<" "
            <<blk.m_offset<<std::endl;*/
        return blk;
    }

public:

    const int m_m; // number of rows
    const int m_n; // number of columns

    DistributedBlockMatrix(const int m, const int n, const char*fileName)
    : m_block(get_block_info(m, n)), m_m(m), m_n(n)
    {
        MPI_Alloc_mem(
            m_block.m_number_of_columns*m_block.m_number_of_rows*sizeof(float), 
            MPI_INFO_NULL, 
            &m_ptr
        );
        MPI_File_open(
            MPI_COMM_WORLD, fileName, 
            MPI_MODE_CREATE | MPI_MODE_RDWR,
            MPI_INFO_NULL, &m_file
        );
        for(int r=0; r<m_block.m_number_of_rows; ++r) 
        {
            MPI_File_seek(
                m_file, sizeof(float)*(m_block.m_offset+r*m_n), MPI_SEEK_SET
            );
            MPI_File_read(
                m_file, 
                m_ptr+r*m_block.m_number_of_columns, 
                m_block.m_number_of_columns, 
                MPI_FLOAT, 
                MPI_STATUS_IGNORE
            );
        }
    }

    // hidden fonctions
    DistributedBlockMatrix() = delete;
    DistributedBlockMatrix(DistributedBlockMatrix&) = delete;
    DistributedBlockMatrix operator=(DistributedBlockMatrix&) = delete;

    ~DistributedBlockMatrix()
    {
        Synchronize();
        MPI_Free_mem(m_ptr);
        MPI_File_close(&m_file);
    }

    // force write to the "distributed" file
    void Synchronize()
    {
        for(int r=0; r<m_block.m_number_of_rows; ++r)
        {
            MPI_File_seek(
                m_file, sizeof(float)*(m_block.m_offset+r*m_n), MPI_SEEK_SET
            );
            MPI_File_write(
                m_file, 
                m_ptr+r*m_block.m_number_of_columns, 
                m_block.m_number_of_columns, 
                MPI_FLOAT, 
                MPI_STATUS_IGNORE
            );
        }
        MPI_File_sync(m_file);
    }

    // return the first row index
    int Start() const { return m_block.m_first_row; }

    // return the first bad row after local values
    int End() const { return m_block.m_first_row+m_block.m_number_of_rows; }

    // access to a row of a DistributedRowMatrix
    class MatrixRow {
        const unsigned m_width;
        const unsigned m_first;
        float*m_v;
        MatrixRow();
    public:
        MatrixRow(const unsigned row, DistributedBlockMatrix&m) :
            m_width(m.m_block.m_number_of_columns), m_first(m.m_block.m_first_column), 
            m_v(m.m_ptr+(row-m.m_block.m_first_row)*m.m_block.m_number_of_columns)
        {}
        float&operator[](const int col) {
            if( col<Start() || col>=End() )
                throw MatrixException( "Matrix: column out-of-bound access" );
            return m_v[col-m_first];
        }
        // return the first col index
        int Start() const { return m_first; }
        // return the first bad col after local values
        int End() const { return m_first+m_width; }
    };

    MatrixRow operator[](const int row) {
        if( row<Start() || row>=End() )
            throw MatrixException( "Matrix: row out-of-bound access" );
        return MatrixRow(row,*this);
    }

    class ConstMatrixRow {
        const unsigned m_width;
        const unsigned m_first;
        const float*const m_v;
        ConstMatrixRow();
    public:
        ConstMatrixRow(const unsigned row, const DistributedBlockMatrix&m) :
            m_width(m.m_block.m_number_of_columns), m_first(m.m_block.m_first_column), 
            m_v(m.m_ptr+(row-m.m_block.m_first_row)*m.m_block.m_number_of_columns)
        {}
        float operator[](const int col) const {
            if( col<Start() || col>=End() )
                throw MatrixException( "Matrix: column out-of-bound access" );
            return m_v[col-m_first];
        }
        // return the first col index
        int Start() const { return m_first; }
        // return the first bad col after local values
        int End() const { return m_first+m_width; }
    };

    ConstMatrixRow operator[](const int row) const {
        if( row<Start() || row>=End() )
            throw MatrixException( "Matrix: row out-of-bound access" );
        return ConstMatrixRow(row,*this);
    }
};

#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <fcntl.h>
#include <cmath>
#if defined(WIN32) || defined(_MSC_VER)
# include <io.h>
# define CLOSE _close
#else 
# include <sys/types.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <unistd.h>
#define CLOSE close
#endif
#include "Transposition.h"

namespace
{

    bool errors_on_check(const std::string& msg, const int d0, const int d1, const float *p0, const float *p1) {
        std::cout << msg << std::endl;
        if( d0>=0 ) CLOSE( d0 );
        if( d1>=0 ) CLOSE( d1 );
        if( p0 != nullptr ) delete p0;
        if( p1 != nullptr ) delete p1;
        return false;
    }

#if defined(WIN32) || defined(_MSC_VER)
    bool check(
        const int N,
        const char *matrixAFileName,
        const char *matrixBFileName
    ) {
        // open file, check the results
        const int dA = _sopen(matrixAFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dA == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, nullptr, nullptr);
        const int dB = _sopen(matrixBFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dB == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixBFileName<<" ... ").str(), dA, -1, nullptr, nullptr);
        float *A = new float[N*N], *B = new float[N];
        const int sizeFull = sizeof(float) * N * N;
        if (_read(dA, A, sizeFull) != sizeFull)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix A file is too small ... unable to read N*N floats")).str(), dA, dB, A, B);
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (_read(dB, B, sizeRow) != sizeRow)
                return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix B file is too small ... unable to get row ")<<r).str(), dA, dB, A, B);
            for (int c = 0; c < N; ++c) {
                //std::cout << std::setfill('0') << std::setw(3) << B[c] << " ";
                if (B[c] != A[c*N+r]) 
                    return errors_on_check( 
                        static_cast<std::ostringstream&&>(std::ostringstream()<<"Your transposition is not correct"<<std::endl
                        <<" -> row="<<r<<", col="<<c).str(), 
                        dA, dB, A, B
                    );
            }
            //std::cout << std::endl;
        }
        delete A;
        delete B;
        CLOSE(dA);
        CLOSE(dB);
        return true;
    }

    void printMatrix(
        const int N,
        const char *matrixAFileName
    ) {
        // open file, check the results
        const int dA = _sopen(matrixAFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dA == -1)
        {
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, nullptr, nullptr);
            return;
        }
        float *A = new float[N];
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (_read(dA, A, sizeRow) != sizeRow)
            {
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix file is too small ... unable to get row ")<<r).str(), dA, -1, A, nullptr);
                 return;
            }
            for (int c = 0; c < N; ++c) {
                std::cout << std::setfill('0') << std::setw(3) << A[c] << " ";
            }
            std::cout << std::endl;
        }
        delete A;
        CLOSE(dA);
    }
#else
    bool check(
        const int N,
        const char *matrixAFileName,
        const char *matrixBFileName
    ) {
        // open file, check the results
        const int dA = open(matrixAFileName, O_RDONLY);
        if (dA == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, nullptr, nullptr);
        const int dB = open(matrixBFileName, O_RDONLY);
        if (dB == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixBFileName<<" ... ").str(), dA, -1, nullptr, nullptr);
        float *A = new float[N*N], *B = new float[N];
        const int sizeFull = sizeof(float) * N * N;
        if (read(dA, A, sizeFull) != sizeFull)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix A file is too small ... unable to read N*N floats")).str(), dA, dB, A, B);
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (read(dB, B, sizeRow) != sizeRow)
                return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix B file is too small ... unable to get row ")<<r).str(), dA, dB, A, B);
            for (int c = 0; c < N; ++c) {
                if (B[c] != A[c*N+r]) 
                    return errors_on_check( 
                        static_cast<std::ostringstream&&>(std::ostringstream()<<"Your transposition is not correct"<<std::endl
                        <<" -> row="<<r<<", col="<<c).str(), 
                        dA, dB, A, B
                    );
            }
        }
        delete A;
        delete B;
        CLOSE(dA);
        CLOSE(dB);
        return true;
    }
    
    void printMatrix(
        const int N,
        const char *matrixAFileName
    ) {
        // open file, check the results
        const int dA = open(matrixAFileName, O_RDONLY);
        if (dA == -1)
        {
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, nullptr, nullptr);
            return;
        }
        float *A = new float[N];
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (read(dA, A, sizeRow) != sizeRow) {
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix file is too small ... unable to get row ")<<r).str(), dA, -1, A, nullptr);
                return ;
            }
            for (int c = 0; c < N; ++c) {
                std::cout << std::setfill('0') << std::setw(3) << A[c] << " ";
            }
            std::cout << std::endl;
        }
        delete A;
        CLOSE(dA);
    }   
#endif

    // creates two Matrices, then calculates the transposition of the first
    void calculateTranspose(
        const int N,
        const char *const matrixAFileName,
        const char *const matrixBFileName,
        const int P
    ) {
        const OPP::MPI::Torus& torus=OPP::MPI::Torus::build(MPI_COMM_WORLD);
        //creates the matrices ...
        const int rank = torus.getRowRing().getRank() + torus.getRowRing().getSize() * torus.getColumnRing().getRank();
        if( rank == 0 )
            std::cout << "Build matrix A(" << N << "," << N << ") ..." << std::endl;
        DistributedBlockMatrix A(torus, N, N, matrixAFileName);
        for (int r = A.Start(); r < A.End(); ++r)
        {
            DistributedBlockMatrix::MatrixRow row = A[r];
            for (int c = row.Start(); c < row.End(); ++c)
                row[c] = static_cast<float>(r*N+c);
        }
        if( rank == 0 )
            std::cout << "Build matrix B(" << N << "," << N << ") ..." << std::endl;
        DistributedBlockMatrix B(torus, N, N, matrixBFileName);
        for (int r = B.Start(); r < B.End(); ++r)
        {
            DistributedBlockMatrix::MatrixRow row = B[r];
            for (int c = row.Start(); c < row.End(); ++c)
                row[c] = static_cast<float>(rank);
        }
        // do the calculation
        MPI_Barrier( torus.getColumnRing().getComm() );
        if( rank == 0 )
            std::cout << "Call student function ..." << std::endl;
        Transposition(torus, A, B, N, P);
    }

    void do_check(
        const int N,
        const char *const matrixAFileName,
        const char *const matrixBFileName
    ) {        
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if( rank != 0 ) return;
        
        std::cout<<"Check the result ..."<<std::endl;    
        if (check(N, matrixAFileName, matrixBFileName))
            std::cout << "Well done, your software seems to be correct!" << std::endl;
        else
            std::cout << "Your result is uncorrect ... try again!" << std::endl;

    }
} // namespace

// entry function
int main(int argc, char **argv)
{
    const int provided = OPP::MPI::Initializer::init(&argc, &argv, MPI_THREAD_MULTIPLE);
    if( provided != MPI_THREAD_MULTIPLE )
        std::cerr << "Problem initializing MPI with required thread level ..." << std::endl;

    // get the world rank
    const OPP::MPI::Communicator communicator(MPI_COMM_WORLD);
    
    // NB : world_size must be a square number
    const int P = int(round(sqrtf(float(communicator.size))));
    if( P*P != communicator.size )
    {
        if( communicator.rank == 0 )
            std::cerr<<"Error: Number of processes must be a square number PxP (e.g. 4, 9, 16, 25, 36, 49 or 64 ...)."<<std::endl;
        OPP::MPI::Initializer::close();
        return -1;    
    }

    int e = 10;
    const char *matrixAFileName = "matrix.bin";
    const char *matrixTFileName = "matrixTransposed.bin";
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp("-e", argv[i]))
        {
            int value;
            if (i+1<argc && sscanf(argv[i + 1], "%d", &value) == 1)
            {
                e = std::max(3, std::min(value, 32));
                i++;
            }
        }
        else if (!strcmp("-m", argv[i]) && i+1<argc && argv[i + 1][0] != '-')
        {
            matrixAFileName = argv[++i];
        }
        else if (!strcmp("-t", argv[i]) && i+1<argc && argv[i + 1][0] != '-')
        {
            matrixTFileName = argv[++i];
        }
        else if (!strcmp("-h", argv[i]))
        {
            std::cout << "Usage:" << std::endl;
            std::cout << "\t-e <e>: log2(n), for vector of size n and matrix (n x n)" << std::endl;
            std::cout << "\t-m <f>: set the filename for the matrix to transpose" << std::endl;
            std::cout << "\t-t <f>: set the filename for the transposed matrix" << std::endl;
        }
    }

    const int N = 1 << e;

    calculateTranspose(N, matrixAFileName, matrixTFileName, P);

    MPI_Barrier( communicator.communicator );
    
    do_check(N, matrixAFileName, matrixTFileName);

    MPI_Barrier( communicator.communicator );

    if( communicator.rank == 0 && N <= 32 )
        printMatrix( N, matrixTFileName );

    OPP::MPI::Initializer::close();

    return 0;
}

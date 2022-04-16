#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
#include <utils/chronoCPU.h>
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
#define CLOSE _close
#else 
# include <unistd.h>
#define CLOSE close
#endif
#include "Produit.h"

namespace
{

    bool errors_on_check(const std::string& msg, const int d0, const int d1, const int d2, const float *p0, const float *p1, const float *p2) {
        std::cout << msg << std::endl;
        if( d0>=0 ) CLOSE( d0 );
        if( d1>=0 ) CLOSE( d1 );
        if( d2>=0 ) CLOSE( d2 );
        if( p0 != nullptr ) delete p0;
        if( p1 != nullptr ) delete p1;
        if( p2 != nullptr ) delete p2;
        return false;
    }

#if defined(WIN32) || defined(_MSC_VER)
    bool check(
        const int N,
        const char *matrixAFileName,
        const char *matrixBFileName,
        const char *matrixCFileName
    ) {
        // open file, check the results
        const int dA = _sopen(matrixAFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dA == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, -1, nullptr, nullptr, nullptr);
        const int dB = _sopen(matrixBFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dB == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixBFileName<<" ... ").str(), dA, -1, -1, nullptr, nullptr, nullptr);
        const int dC = _sopen(matrixCFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dC == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixCFileName<<" ... ").str(), dA, dB, -1, nullptr, nullptr, nullptr);
        float *A = new float[N], *B = new float[N*N], *C = new float[N];
        const int sizeFull = sizeof(float) * N * N;
        if (_read(dB, B, sizeFull) != sizeFull)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix B file is too small ... unable to read N*N floats")).str(), dA, dB, dC, A, B, C);
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (_read(dA, A, sizeRow) != sizeRow)
                return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix A file is too small ... unable to get row")).str(), dA, dB, dC, A, B, C);
            if (_read(dC, C, sizeRow) != sizeRow)
                return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix C file is too small ... unable to get row ")<<r).str(), dA, dB, dC, A, B, C);
            for (int c = 0; c < N; ++c) {
                float p = 0.f;
                for(int i=0; i<N; ++i) p += A[i] * B[i*N+c];
                if (fabs(p/C[c]-1.f) > 1e-3f) 
                    return errors_on_check( 
                        static_cast<std::ostringstream&&>(std::ostringstream()<<"Your product is uncorrect"<<std::endl
                        <<" -> row="<<r<<", col="<<c<<", wait "<<p<<", get "<<C[c]).str(), 
                        dA, dB, dC, A, B, C
                    );
            }
        }
        delete A;
        delete B;
        delete C;
        _close(dA);
        _close(dB);
        _close(dC);
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
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, -1, nullptr, nullptr, nullptr);
            return;
        }
        float *A = new float[N];
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (_read(dA, A, sizeRow) != sizeRow)
            {
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix file is too small ... unable to get row ")<<r).str(), dA, -1, -1, A, nullptr, nullptr);
                 return;
            }
            for (int c = 0; c < N; ++c) {
                std::cout << std::setfill('0') << std::setw(3) << A[c] << " ";
            }
            std::cout << std::endl;
        }
        delete A;
        _close(dA);
    }
#else
    bool check(
        const int N,
        const char *matrixAFileName,
        const char *matrixBFileName,
        const char *matrixCFileName
    ) {
        // open file, check the results
        const int dA = open(matrixAFileName, O_RDONLY);
        if (dA == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, -1, nullptr, nullptr, nullptr);
        const int dB = open(matrixBFileName, O_RDONLY);
        if (dB == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixBFileName<<" ... ").str(), dA, -1, -1, nullptr, nullptr, nullptr);
        const int dC = open(matrixCFileName, O_RDONLY);
        if (dC == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixCFileName<<" ... ").str(), dA, dB, -1, nullptr, nullptr, nullptr);
        float *A = new float[N], *B = new float[N*N], *C = new float[N];
        const int sizeFull = sizeof(float) * N * N;
        if (read(dB, B, sizeFull) != sizeFull)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix B file is too small ... unable to read N*N floats")).str(), dA, dB, dC, A, B, C);
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (read(dA, A, sizeRow) != sizeRow)
                return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix A file is too small ... unable to get row")).str(), dA, dB, dC, A, B, C);
            if (read(dC, C, sizeRow) != sizeRow)
                return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix C file is too small ... unable to get row ")<<r).str(), dA, dB, dC, A, B, C);
            for (int c = 0; c < N; ++c) {
                float p = 0.f;
                for(int i=0; i<N; ++i) p += A[i] * B[i*N+c];
                if (fabs(p/C[c]-1.f) > 1e-3f) 
                    return errors_on_check( 
                        static_cast<std::ostringstream&&>(std::ostringstream()<<"Your product is uncorrect"<<std::endl
                        <<" -> row="<<r<<", col="<<c<<", wait "<<p<<", get "<<C[c]).str(), 
                        dA, dB, dC, A, B, C
                    );
            }
        }
        delete A;
        delete B;
        delete C;
        close(dA);
        close(dB);
        close(dC);
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
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixAFileName<<" ... ").str(), -1, -1, -1, nullptr, nullptr, nullptr);
            return;
        }
        float *A = new float[N];
        const int sizeRow = sizeof(float) * N;
        for (int r = 0; r < N; ++r)
        {
            if (read(dA, A, sizeRow) != sizeRow) {
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix file is too small ... unable to get row ")<<r).str(), dA, -1, -1, A, nullptr, nullptr);
                return ;
            }
            for (int c = 0; c < N; ++c) {
                std::cout << std::setfill('0') << std::setw(3) << A[c] << " ";
            }
            std::cout << std::endl;
        }
        delete A;
        close(dA);
    }   
#endif

    // creates three Matrices, then calculates the product of the first two
    void doProduct(
        const int N,
        const char *const matrixAFileName,
        const char *const matrixBFileName,
        const char *const matrixCFileName,
        const int P
    ) {
        //creates the matrices ...
        const OPP::MPI::Communicator communicator(MPI_COMM_WORLD);
        if( communicator.rank == 0 )
            std::cout << "Build matrix A(" << N << "," << N << ") ..." << std::endl;
        const OPP::MPI::Torus torus = OPP::MPI::Torus::build(communicator.communicator);
        DistributedBlockMatrix A(torus, N, N, matrixAFileName);
        for (int r = A.Start(); r < A.End(); ++r)
        {
            DistributedBlockMatrix::MatrixRow row = A[r];
            for (int c = row.Start(); c < row.End(); ++c)
                row[c] = static_cast<float>(r+1);
        }
        if( communicator.rank == 0 )
            std::cout << "Build matrix B(" << N << "," << N << ") ..." << std::endl;
        DistributedBlockMatrix B(torus, N, N, matrixBFileName);
        for (int r = B.Start(); r < B.End(); ++r)
        {
            DistributedBlockMatrix::MatrixRow row = B[r];
            for (int c = row.Start(); c < row.End(); ++c)
                row[c] = static_cast<float>(c+1);
        }
        DistributedBlockMatrix C(torus, N, N, matrixCFileName);
        MPI_Barrier( communicator.communicator );
        // do the calculation
        if( communicator.rank == 0 )
            std::cout << "Call student function ..." << std::endl;
        ChronoCPU chr;
        chr.start();
        Produit(torus, A, B, C);
        MPI_Barrier( MPI_COMM_WORLD );
        chr.stop();
        if( communicator.rank == 0)
            std::cout << "Elapsed Time: "<<chr.elapsedTime()<<" ms"<<std::endl;
    }

    void do_check(
        const int N,
        const char *const matrixAFileName,
        const char *const matrixBFileName,
        const char *const matrixCFileName
    ) {        
        const OPP::MPI::Communicator communicator(MPI_COMM_WORLD);
        if( communicator.rank != 0 ) return;
        
        std::cout<<"Check the result ..."<<std::endl;    
        if (check(N, matrixAFileName, matrixBFileName, matrixCFileName))
            std::cout << "Well done, your software seems to be correct!" << std::endl;
        else
            std::cout << "Your result is uncorrect ... try again!" << std::endl;

    }
} // namespace

// entry function
int main(int argc, char **argv)
{
    const int provided = OPP::MPI::Initializer::init(&argc, &argv, MPI_THREAD_MULTIPLE);

    // get the world rank
    const OPP::MPI::Communicator communicator(MPI_COMM_WORLD);
    
    // NB : world_size must be a square number
    const int P = int(round(sqrtf(float(communicator.size))));
    if( P*P != communicator.size )
    {
        if( communicator.rank == 0 )
            std::cerr<<"Error: Number of processes must be a square number PxP (e.g. 16 or 64 ...)."<<std::endl;
        MPI_Finalize();
        return -1;    
    }

    int e = 10;
    const char *matrixAFileName = "matrixA.bin";
    const char *matrixBFileName = "matrixB.bin";
    const char *matrixCFileName = "matrixC.bin";
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp("-e", argv[i]))
        {
            int value;
            if (i+1<argc && sscanf(argv[i + 1], "%d", &value) == 1)
            {
                if( value < 3 ) e = 3;
                else if ( value > 32 ) e = 32;
                else e = value;
                i++;
            }
        }
        else if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i]))
        {
            std::cout << "Usage:" << std::endl;
            std::cout << "\t-e <e>: log2(n), for vector of size n and matrix (n x n)" << std::endl;
        }
    }

    const int N = 1 << e;

    doProduct(N, matrixAFileName, matrixBFileName, matrixCFileName, P);

    MPI_Barrier( communicator.communicator );
    
    do_check(N, matrixAFileName, matrixBFileName, matrixCFileName);

    MPI_Barrier( communicator.communicator );

    if( communicator.rank == 0 && N <= 32 )
        printMatrix( N, matrixCFileName );

    OPP::MPI::Initializer::close();

    return 0;
}

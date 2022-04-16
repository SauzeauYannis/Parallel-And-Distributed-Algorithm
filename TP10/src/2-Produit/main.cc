#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <fcntl.h>
#include <cmath>
#ifdef WIN32
#include <io.h>
#define CLOSE _close
#else 
#include <unistd.h>
#define CLOSE close
#endif
#include <MPI/OPP_MPI.h>
#include <2-Produit/Produit.h>

namespace
{

    bool errors_on_check(const std::string& msg, const int d0, const int d1, const int d2, const float *p0, const float *p1) {
        std::cout << msg << std::endl;
        if( d0>=0 ) CLOSE( d0 );
        if( d1>=0 ) CLOSE( d1 );
        if( d2>=0 ) CLOSE( d2 );
        if( p0 != nullptr ) delete p0;
        if( p1 != nullptr ) delete p1;
        return false;
    }

#ifdef WIN32
    bool check(
        const OPP::MPI::Communicator& communicator,
        const int N,
        const char *vectorFileName,
        const char *matrixFileName,
        const char *resultFileName)
    {
        // open file, check the results
        const int dv = _sopen(vectorFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dv == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<vectorFileName<<" ... ").str(), -1, -1, -1, nullptr, nullptr);
        const int dm = _sopen(matrixFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dm == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixFileName<<" ... ").str(), dv, -1, -1, nullptr, nullptr);
        const int dr = _sopen(resultFileName, _O_BINARY | _O_RDONLY, _SH_DENYWR);
        if (dr == -1)
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<resultFileName<<" ... ").str(), dv, dm, -1, nullptr, nullptr);
        float *A = new float[N], *B = new float[N];
        const int size = sizeof(float) * N;
        if (_read(dv, B, size) != size)
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("vector file is too small ... unable to read N floats")).str(), dv, dm, dr, A, B);
        for (int r = 0; r < N; ++r)
        {
            if (_read(dm, A, size) != size)
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix file is too small ... unable to get row ")<<r).str(), dv, dm, dr, A, B);
            float truth = 0.f;
            for (int c = 0; c < N; ++c)
                truth += A[c] * B[c];
            float result;
            if (_read(dr, &result, sizeof(float)) != sizeof(float))
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to read result vector at position ")<<r).str(), dv, dm, dr, A, B);
            if (fabs(truth - result) > 1e-3f)
                errors_on_check( 
                    static_cast<std::ostringstream&&>(std::ostringstream("Your calculation is far too different to the good result")<<std::endl
                    <<" -> row="<<r<<", truth="<<truth<<", result="<<result).str(), 
                    dv, dm, dr, A, B
                );
        }
        delete A;
        delete B;
        _close(dv);
        _close(dm);
        _close(dr);
        return true;
    }
#else
    bool check(
        const OPP::MPI::Communicator& communicator,
        const int N,
        const char *vectorFileName,
        const char *matrixFileName,
        const char *resultFileName)
    {
        // open file, check the results
        const int dv = open(vectorFileName, O_RDONLY);
        if (dv == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<vectorFileName<<" ... ").str(), -1, -1, -1, nullptr, nullptr);
        const int dm = open(matrixFileName, O_RDONLY);
        if (dm == -1)
            return errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<matrixFileName<<" ... ").str(), dv, -1, -1, nullptr, nullptr);
        const int dr = open(resultFileName, O_RDONLY);
        if (dr == -1)
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to open ")<<resultFileName<<" ... ").str(), dv, dm, -1, nullptr, nullptr);
        float *A = new float[N], *B = new float[N];
        const int size = sizeof(float) * N;
        if (read(dv, B, size) != size)
            errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("vector file is too small ... unable to read N floats")).str(), dv, dm, dr, A, B);
        for (int r = 0; r < N; ++r)
        {
            if (read(dm, A, size) != size)
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("matrix file is too small ... unable to get row ")<<r).str(), dv, dm, dr, A, B);
            float truth = 0.f;
            for (int c = 0; c < N; ++c)
                truth += A[c] * B[c];
            float result;
            if (read(dr, &result, sizeof(float)) != sizeof(float))
                errors_on_check( static_cast<std::ostringstream&&>(std::ostringstream("Unable to read result vector at position ")<<r).str(), dv, dm, dr, A, B);
            if (fabs(truth - result) > 1e-3f)
                errors_on_check( 
                    static_cast<std::ostringstream&&>(std::ostringstream("Your calculation is far too different to the good result")<<std::endl
                    <<" -> row="<<r<<", truth="<<truth<<", result="<<result).str(), 
                    dv, dm, dr, A, B
                );
        }
        delete A;
        delete B;
        CLOSE(dv);
        CLOSE(dm);
        CLOSE(dr);
        return true;
    }
#endif

    // creates a Matrix and a Vector, then do their product
    void calculateProduct(
        const OPP::MPI::Communicator communicator, 
        const int N,
        const char *const vectorFileName,
        const char *const matrixFileName,
        const char *const resultFileName
    ) {
        //creates the vector and matrix ...
        if( communicator.rank == 0 )
            std::cout << "Build matrix A(" << N << "," << N << ") ..." << std::endl;
        DistributedRowMatrix A(communicator, N, N, matrixFileName);
        for (int r = A.Start(); r < A.End(); ++r)
        {
            DistributedRowMatrix::MatrixRow row = A[r];
            for (int c = 0; c < N; ++c)
                row[c] = static_cast<float>(r + 1);
        }
        if( communicator.communicator == 0 )
            std::cout << "Build vector B(" << N << ") ..." << std::endl;
        DistributedBlockVector B(communicator, N, vectorFileName);
        for (int i = B.Start(); i < B.End(); ++i)
            B[i] = 1.f;
        // do the product
        if( communicator.rank == 0 )
            std::cout << "Do the product ..." << std::endl;
        DistributedBlockVector X(communicator, N, resultFileName);
        MPI_Barrier( MPI_COMM_WORLD );
        produit(communicator, A, B, X, N);
    }

    void do_check(
        const OPP::MPI::Communicator communicator, 
        const int N,
        const char *const vectorFileName,
        const char *const matrixFileName,
        const char *const resultFileName
    ) {        
        
        if( communicator.rank != 0 ) return;
        
        if (check(communicator, N, vectorFileName, matrixFileName, resultFileName))
            std::cout << "Well done, your software seems to be correct!" << std::endl;
        else
            std::cout << "Your result is uncorrect ... try again!" << std::endl;

    }


    void display_info(const OPP::MPI::Communicator& communicator)
    {
        if( communicator.rank > 0 )
            return ;

        std::cout << "*** Number of Processors: " << communicator.size << " ***" << std::endl;
    }
} // namespace

// entry function
int main(int argc, char **argv)
{
    OPP::MPI::Initializer::init(&argc, &argv);
    const OPP::MPI::Communicator communicator(MPI_COMM_WORLD);

    display_info(communicator);

    int e = 10;
    const char *vectorFileName = "vector.bin";
    const char *resultFileName = "result.bin";
    const char *matrixFileName = "matrix.bin";
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
        else if (!strcmp("-v", argv[i]) && i+1<argc && argv[i + 1][0] != '-')
        {
            vectorFileName = argv[++i];
        }
        else if (!strcmp("-m", argv[i]) && i+1<argc && argv[i + 1][0] != '-')
        {
            matrixFileName = argv[++i];
        }
        else if (!strcmp("-x", argv[i]) && i+1<argc && argv[i + 1][0] != '-')
        {
            resultFileName = argv[++i];
        }
        else if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i]))
        {
            std::cout << "Usage:" << std::endl;
            std::cout << "\t-e <e>: log2(n), for vector of size n and matrix (n x n)" << std::endl;
            std::cout << "\t-v <f>: set the filename for the vector" << std::endl;
            std::cout << "\t-m <f>: set the filename for the matrix" << std::endl;
            std::cout << "\t-x <f>: set the filename for the resulting vector" << std::endl;
        }
    }

    const int N = 1 << e;

    if( (N % communicator.size) != 0 ) {
        if( communicator.rank == 0 )
        {
            std::cerr << "N is not a multiple of the number of MPI processors!" << std::endl;
            std::cerr << "Abort()" << std::endl;
        }
    } 
    else {
        calculateProduct(communicator, N, vectorFileName, matrixFileName, resultFileName);

        MPI_Barrier( communicator.communicator );

        do_check(communicator, N, vectorFileName, matrixFileName, resultFileName);
    }

    OPP::MPI::Initializer::close();

    return 0;
}

#define _CRT_SECURE_NO_WARNINGS 1
#include <iostream>
#include <utils/DistributedBlockVector.h>
#include <utils/DistributedRowMatrix.h>
#include <algorithm>
#include <1-Resolution/Resolution.h>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <fcntl.h>
#include <cmath>
#ifdef WIN32
# include <io.h>
namespace {
    int __cdecl CLOSE(int fd) { 
        return _close(fd); 
    }
    int READ(int fd, void*buffer, unsigned int max) {
        return _read(fd, buffer, max);
    }
    int SOPEN(
        char const* const _FileName,
        int         const _OFlag,
        int         const _ShFlag,
        int         const _PMode = 0
    ) {
        return _sopen(_FileName, _OFlag, _ShFlag, _PMode);
    }
}
# define O_BINARY _O_BINARY
# define O_RDONLY _O_RDONLY
# define SH_DENYWR _SH_DENYWR
#else 
# include <unistd.h>
namespace{
    int CLOSE(int fd) { return close(fd); }
    int READ(int fd, void*buffer, unsigned int max) {
        return read(fd, buffer, max);
    }
    int SOPEN(
        char const* const _FileName,
        int         const _OFlag,
        int         const _ShFlag,
        int         const _PMode = 0
    ) {
        return open(_FileName, _OFlag, _ShFlag, _PMode);
    }
}
# define O_BINARY 0
# define SH_DENYWR 0644
#endif


// creates a Matrix and a Vector, then do their product
void do_solve(
    const OPP::MPI::Communicator communicator, 
    const int N,
    const char *const vectorFileName,
    const char *const matrixFileName,
    const char *const resultFileName
) {
    //creates the vector and matrix ...
    if( communicator.rank == 0 )
        std::cout << "Build matrix L(" << N << "," << N << ") ..." << std::endl;
    DistributedRowMatrix L(communicator, N, N, matrixFileName);
    for (int r = L.Start(); r < L.End(); ++r)
    {
        DistributedRowMatrix::MatrixRow row = L[r];
        for (int c = 0; c < N; ++c)
            row[c] = static_cast<float>(c<=r ? 1 : 0);
    }
    if( communicator.rank == 0 )
        std::cout << "Build vector B(" << N << ") ..." << std::endl;
    DistributedBlockVector B(communicator, N, vectorFileName);
    for (int i = B.Start(); i < B.End(); ++i) {
        B[i] = 0.f;
        for(int j=0; j<=i; ++j)
            B[i] += L[i][j]*static_cast<float>(j+1);
    }
    // do the product
    DistributedBlockVector X(communicator, N, resultFileName);
    MPI_Barrier( MPI_COMM_WORLD );
    if( communicator.rank == 0 )
        std::cout << "Solve the system ..." << std::endl;
    Solve(communicator, L, B, X, N);
}

void do_check(
    const OPP::MPI::Communicator communicator, 
    const int N, 
    const char *const vectorFileName,
    const char *const matrixFileName,
    const char *const resultFileName 
) {
    if( communicator.rank > 0 ) return;
    std::cout << "check the result ..."<<std::endl;
    // open file, check the results
    const int dl = SOPEN(matrixFileName, O_BINARY | O_RDONLY, SH_DENYWR);
    const int db = SOPEN(vectorFileName, O_BINARY | O_RDONLY, SH_DENYWR);
    const int dr = SOPEN(resultFileName, O_BINARY | O_RDONLY, SH_DENYWR);
    if (dl == -1) 
    {
        std::cerr << "Impossible d'ouvrir "<<matrixFileName<<std::endl;
        return ;
    }
    if (db == -1) 
    {
        std::cerr << "Impossible d'ouvrir "<<vectorFileName<<std::endl;
        return ;
    }
    if (dr == -1) 
    {
        std::cerr << "Impossible d'ouvrir "<<resultFileName<<std::endl;
        return ;
    }
    float *B = new float[N];
    const int size = sizeof(float) * N;
    if (READ(db, B, size) != size)
    {
        std::cerr << "Impossible de lire le vecteur B" << std::endl;
        return ;
    }
    float *X = new float[N];
    if (READ(dr, X, size) != size)
    {
        std::cerr << "Impossible de lire le vecteur résultat" << std::endl;
        return ;
    }
    float *L = new float[N];
    for (int r = 0; r < N; ++r)
    {
        if (READ(dl, L, size) != size)
        {
            std::cerr << "Impossible de lire la matrice" << std::endl;
            return ;
        }
        // on calcule (mauvais schéma ... il faudrait sum = 0.f !)
        float sum = B[r];
        for(int i=0; i<r; ++i)
            sum -= X[i] * L[i];
        float Xi = sum / L[r];
        if( fabs(X[r]-Xi) > 1e-2f )
        {
            std::cerr <<"Mauvaise valeur en "<<r<<". Attendue "<<Xi<<", lue "<<X[r]<<std::endl;
            return;
        }
    }
    std::cout << "Well done, it seems to work!" << std::endl;
    delete L;
    delete B;
    delete X;
    CLOSE(dl);
    CLOSE(db);
    CLOSE(dr);
}

void display_info(const OPP::MPI::Communicator& communicator)
{
    if( communicator.rank > 0 )
        return ;

    std::cout << "*** Number of Processors: " << communicator.size << " ***" << std::endl;
}


// entry function
int main(int argc, char **argv)
{
    OPP::MPI::Initializer::init(&argc, &argv);

    const OPP::MPI::Communicator communicator(MPI_COMM_WORLD);
    
    int e = 10;
    const char *vectorFileName = "vector.bin";
    const char *resultFileName = "result.bin";
    const char *matrixFileName = "matrix.bin";
    for (int i = 0; i < argc - 1; ++i)
    {
        if (!strcmp("-e", argv[i]))
        {
            int value;
            if (sscanf(argv[i + 1], "%d", &value) == 1)
            {
                e = std::max(2, std::min(value, 32));
                i++;
            }
        }
        else if (!strcmp("-v", argv[i]) && argv[i + 1][0] != '-')
        {
            vectorFileName = argv[++i];
        }
        else if (!strcmp("-m", argv[i]) && argv[i + 1][0] != '-')
        {
            matrixFileName = argv[++i];
        }
        else if (!strcmp("-x", argv[i]) && argv[i + 1][0] != '-')
        {
            resultFileName = argv[++i];
        }
        else if (!strcmp("-h", argv[i]))
        {
            std::cout << "Usage:" << std::endl;
            std::cout << "\t-e <e>: log2(n), for vector of size n and matrix (n x n)" << std::endl;
            std::cout << "\t-v <f>: set the filename for the vector" << std::endl;
            std::cout << "\t-m <f>: set the filename for the matrix" << std::endl;
            std::cout << "\t-x <f>: set the filename for the resulting vector" << std::endl;
        }
    }

    const int N = 1 << e;

    display_info(communicator);

    do_solve(communicator, N, vectorFileName, matrixFileName, resultFileName);

    MPI_Barrier( MPI_COMM_WORLD );

    do_check(communicator, N, vectorFileName, matrixFileName, resultFileName);

    MPI_Barrier( MPI_COMM_WORLD );

    OPP::MPI::Initializer::close();
    return 0;
}

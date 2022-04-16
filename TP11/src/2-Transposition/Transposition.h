#pragma once

#include <DistributedBlockMatrix.h>
#include <MPI/OPP_MPI.h>

/* Effectuer le calcul B = transposée(A) ... */
void Transposition(
    const OPP::MPI::Torus& torus,
    const DistributedBlockMatrix& A,
    DistributedBlockMatrix&B,
    const int N,
    const int P
);

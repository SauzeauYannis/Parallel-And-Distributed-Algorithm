#pragma once

#include <MPI/OPP_MPI.h>
#include <DistributedBlockVector.h>
#include <DistributedRowMatrix.h>

void Solve(
    const OPP::MPI::Communicator& communicator,
    const DistributedRowMatrix& L,
    const DistributedBlockVector& B,
    DistributedBlockVector& X,
    const int N
);
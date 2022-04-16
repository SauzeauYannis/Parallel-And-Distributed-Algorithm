#pragma once

#include <DistributedBlockMatrix.h>
#include <MPI/OPP_MPI.h>

void Produit(
    const OPP::MPI::Torus& torus,
    const DistributedBlockMatrix& A,
    const DistributedBlockMatrix& B,
          DistributedBlockMatrix& C
);

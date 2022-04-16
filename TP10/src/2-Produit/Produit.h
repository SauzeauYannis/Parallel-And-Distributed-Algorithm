#pragma once

#include <DistributedBlockVector.h>
#include <DistributedRowMatrix.h>
#include <MPI/OPP_MPI.h>

/* Effectuer le calcul X = A fois B ... */
void produit(
  const OPP::MPI::Communicator& communicator,
  const DistributedRowMatrix& A,
  const DistributedBlockVector&B,
  DistributedBlockVector& X,
  const int N
);

#include <4-Cannon/Produit.h>
#include <DistributedBlockMatrix.h>

#include <memory>

namespace {

void RotationHorizontale(const OPP::MPI::Torus &torus, const int x, const int y,
                         float *buffer, const int L) {
  if (torus.getColumnRing().getRank() == x) {
    if (torus.getRowRing().getRank() == y) {
      torus.Send(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::WEST);
      torus.Recv(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::EAST);
    }
  }
}

void RotationVerticale(const OPP::MPI::Torus &torus, const int x, const int y,
                       float *buffer, const int L) {
  if (torus.getRowRing().getRank() == y) {
    if (torus.getColumnRing().getRank() == x) {
      torus.Send(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::NORTH);
      torus.Recv(buffer, L, MPI_FLOAT, OPP::MPI::Torus::Direction::SOUTH);
    }
  }
}

void ProduitSequentiel(float *A, float *B, DistributedBlockMatrix &C, int n) {
  for (int row = C.Start(); row < C.End(); ++row) {
    for (int col = C[row].Start(); col < C[row].End(); ++col) {
      float dot = 0.0;
      for (int k = 0; k < n; ++k)
        dot += A[k + (row - C.Start()) * n] * B[(col - C[row].Start()) + k * n];
      C[row][col] += dot;
    }
  }
}

}  // namespace

void Produit(const OPP::MPI::Torus &torus, const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B, DistributedBlockMatrix &C) {
  const int n = sqrt(torus.getCommunicator().size);
  const int x = torus.getColumnRing().getRank();
  const int y = torus.getRowRing().getRank();
  const int nb_rows = C.End() - C.Start();
  const int L = nb_rows * (C[C.Start()].End() - C[C.Start()].Start());

  float *bufferA = new float[L];
  float *bufferB = new float[L];

  for (int i = A.Start(); i < A.End(); ++i)
    for (int j = A[i].Start(); j < A[i].End(); ++j)
      bufferA[(j - A[i].Start()) + nb_rows * (i - A.Start())] = A[i][j];

  for (int i = B.Start(); i < B.End(); ++i)
    for (int j = B[i].Start(); j < B[i].End(); ++j)
      bufferB[(j - B[i].Start()) + nb_rows * (i - B.Start())] = B[i][j];

  for (int i = C.Start(); i < C.End(); ++i)
    for (int j = C[i].Start(); j < C[i].End(); ++j) C[i][j] = 0.0f;

  RotationHorizontale(torus, x, y, bufferA, L);
  RotationVerticale(torus, x, y, bufferB, L);

  for (int k = 0; k < n; ++k) {
    ProduitSequentiel(bufferA, bufferB, C, nb_rows);

    RotationHorizontale(torus, x, y, bufferA, L);
    RotationVerticale(torus, x, y, bufferB, L);
  }

  RotationHorizontale(torus, x, y, bufferA, L);
  RotationVerticale(torus, x, y, bufferB, L);
}

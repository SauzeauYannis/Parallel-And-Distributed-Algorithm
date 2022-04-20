#include <6-Snyder/Produit.h>
#include <DistributedBlockMatrix.h>

#include <memory>
#include <thread>

namespace {

void ProduitSequentiel(float *A, float *B, DistributedBlockMatrix &C, int r) {
  for (int row = C.Start(); row < C.End(); ++row) {
    for (int col = C[row].Start(); col < C[row].End(); ++col) {
      float dot = 0.0;
      for (int k = 0; k < r; ++k)
        dot += A[k + (row - C.Start()) * r] * B[(col - C[row].Start()) + k * r];
      C[row][col] += dot;
    }
  }
}

void init(const DistributedBlockMatrix &A, const DistributedBlockMatrix &B,
          DistributedBlockMatrix &C, float *bufferA, float *bufferB,
          const int r) {
  for (int i = A.Start(); i < A.End(); ++i)
    for (int j = A[i].Start(); j < A[i].End(); ++j)
      bufferA[(j - A[i].Start()) + r * (i - A.Start())] = A[i][j];

  for (int i = B.Start(); i < B.End(); ++i)
    for (int j = B[i].Start(); j < B[i].End(); ++j)
      bufferB[(j - B[i].Start()) + r * (i - B.Start())] = B[i][j];

  for (int i = C.Start(); i < C.End(); ++i)
    for (int j = C[i].Start(); j < C[i].End(); ++j) C[i][j] = 0.0f;
}

}  // namespace

void Produit(const OPP::MPI::Torus &torus, const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B, DistributedBlockMatrix &C) {
  const int n = sqrt(torus.getCommunicator().size);
  const int x = torus.getColumnRing().getRank();
  const int y = torus.getRowRing().getRank();
  const int r = C.End() - C.Start();
  const int L = r * r;

  float *bufferA = new float[L];
  float *bufferB = new float[L];
  float *bufferC = new float[L];

  init(A, B, C, bufferA, bufferB, r);
}

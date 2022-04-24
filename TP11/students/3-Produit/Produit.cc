#include <3-Produit/Produit.h>
#include <MPI/OPP_MPI.h>

#include <memory>

namespace {

void BroadcastRow(const OPP::MPI::Torus &torus, const int x, const int k,
                  float *src, float *dest, const int L, const int r) {
  if (torus.getColumnRing().getRank() == x) {
    using Direction = OPP::MPI::Torus::Direction;

    if (torus.getRowRing().getRank() == k) {
      for (int i = 0; i < r; ++i) {
        torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
        for (int j = 0; j < L / r; ++j)
          dest[i * L / r + j] = src[i * L / r + j];
      }
    } else if (torus.getRowRing().getNext() == k) {
      for (int i = 0; i < r; ++i)
        torus.Recv(&dest[i * L / r], L / r, MPI_FLOAT, Direction::WEST);
    } else {
      torus.Recv(dest, L / r, MPI_FLOAT, Direction::WEST);
      for (int i = 0; i < r - 1; ++i) {
        MPI_Request request =
            torus.AsyncSend(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
        torus.Recv(&dest[(i + 1) * L / r], L / r, MPI_FLOAT, Direction::WEST);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
      torus.Send(&src[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::EAST);
    }
  }
}

void BroadcastCol(const OPP::MPI::Torus &torus, const int k, const int y,
                  float *src, float *dest, const int L, const int r) {
  if (torus.getRowRing().getRank() == y) {
    using Direction = OPP::MPI::Torus::Direction;

    if (torus.getColumnRing().getRank() == k) {
      for (int i = 0; i < r; ++i) {
        torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::SOUTH);
        for (int j = 0; j < L / r; ++j)
          dest[i * L / r + j] = src[i * L / r + j];
      }
    } else if (torus.getColumnRing().getNext() == k) {
      for (int i = 0; i < r; ++i)
        torus.Recv(&dest[i * L / r], L / r, MPI_FLOAT, Direction::NORTH);
    } else {
      torus.Recv(dest, L / r, MPI_FLOAT, Direction::NORTH);
      for (int i = 0; i < r - 1; ++i) {
        MPI_Request request = torus.AsyncSend(&src[i * L / r], L / r, MPI_FLOAT,
                                              Direction::SOUTH);
        torus.Recv(&dest[(i + 1) * L / r], L / r, MPI_FLOAT, Direction::NORTH);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
      torus.Send(&src[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::SOUTH);
    }
  }
}

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

  float *send_bufferA = new float[L];
  float *send_bufferB = new float[L];

  init(A, B, C, send_bufferA, send_bufferB, r);

  float *recv_bufferA = new float[L];
  float *recv_bufferB = new float[L];
  
  for (int k = 0; k < n; ++k) {
    BroadcastRow(torus, x, k, send_bufferA, recv_bufferA, L, r);
    BroadcastCol(torus, k, y, send_bufferB, recv_bufferB, L, r);

    ProduitSequentiel(send_bufferA, send_bufferB, C, r);
  }
}

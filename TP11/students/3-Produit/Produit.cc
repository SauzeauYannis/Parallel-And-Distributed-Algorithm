#include <3-Produit/Produit.h>
#include <MPI/OPP_MPI.h>

#include <memory>

namespace {

void BroadcastRow(const OPP::MPI::Torus &torus, const int x, const int k,
                  float *src, float *dest, const int L, const int r) {
  if (torus.getColumnRing().getRank() == x) {
    using Direction = OPP::MPI::Torus::Direction;

    if (torus.getRowRing().getRank() == k) {
      for (int i = 0; i < r; ++i)
        torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::EAST);
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
      torus.Send(&dest[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::EAST);
    }
  }
}

void BroadcastCol(const OPP::MPI::Torus &torus, const int k, const int y,
                  float *src, float *dest, const int L, const int r) {
  if (torus.getRowRing().getRank() == y) {
    using Direction = OPP::MPI::Torus::Direction;

    if (torus.getColumnRing().getRank() == k) {
      for (int i = 0; i < r; ++i)
        torus.Send(&src[i * L / r], L / r, MPI_FLOAT, Direction::SOUTH);
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
      torus.Send(&dest[(r - 1) * L / r], L / r, MPI_FLOAT, Direction::SOUTH);
    }
  }
}

void ProduitSequentiel(float *A, float *B, float *C, int n) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; col++) {
      float dot = 0.0;
      for (int k = 0; k < n; ++k) dot += A[k + row * n] * B[col + k * n];
      C[col + row * n] = dot;
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

  float *send_bufferA = new float[L];
  float *send_bufferB = new float[L];

  for (int i = A.Start(); i < A.End(); ++i)
    for (int j = A[i].Start(); j < A[i].End(); ++j)
      send_bufferA[(j - A[i].Start()) + nb_rows * (i - A.Start())] = A[i][j];

  for (int i = B.Start(); i < B.End(); ++i)
    for (int j = B[i].Start(); j < B[i].End(); ++j)
      send_bufferB[(j - B[i].Start()) + nb_rows * (i - B.Start())] = B[i][j];

  for (int i = C.Start(); i < C.End(); ++i)
    for (int j = C[i].Start(); j < C[i].End(); ++j) C[i][j] = 0.0f;

  float *recv_bufferA = new float[L];
  float *recv_bufferB = new float[L];
  float *recv_bufferC = new float[L];

  for (int k = 0; k < n; ++k) {
    BroadcastRow(torus, x, k, send_bufferA, recv_bufferA, L, nb_rows);
    BroadcastCol(torus, k, y, send_bufferB, recv_bufferB, L, nb_rows);

    if (k == 0) {
      ProduitSequentiel(send_bufferA, send_bufferB, recv_bufferC, nb_rows);
    } else {
      ProduitSequentiel(recv_bufferA, recv_bufferB, recv_bufferC, nb_rows);
    }

    for (int i = C.Start(); i < C.End(); ++i)
      for (int j = C[i].Start(); j < C[i].End(); ++j)
        C[i][j] += recv_bufferC[(j - C[i].Start()) + nb_rows * (i - C.Start())];
  }
}

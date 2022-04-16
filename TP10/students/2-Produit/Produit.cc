#include <2-Produit/Produit.h>
#include <mpi.h>
#include <utils/DistributedBlockVector.h>
#include <utils/DistributedRowMatrix.h>

#include <cstring>
//
/* Effectuer le calcul X = A fois B ... */
void produit(const OPP::MPI::Communicator &communicator,
             const DistributedRowMatrix &A, const DistributedBlockVector &B,
             DistributedBlockVector &X, const int N) {
  OPP::MPI::Ring ring(communicator);

  const int p = ring.getSize();
  const int q = ring.getRank();
  const int r = N / p;

  float *tempSend = new float[r];
  float *tempRecv = new float[r];
  for (int i = X.Start(); i < X.End(); i++) X[i] = 0;
  for (int i = B.Start(), n = 0; i < B.End(); i++, n++) tempSend[n] = B[i];

  for (int step = 0; step < p; ++step) {
    MPI_Request sendRequest = ring.AsyncSend(tempSend, r, MPI_FLOAT);
    MPI_Request recvRequest = ring.AsyncRecv(tempRecv, r, MPI_FLOAT);

    for (int j = 0; j < r; ++j) {
      for (int i = 0; i < r; ++i) {
        int column = i + ((q + p - step) % p) * r;
        X[X.Start() + j] += A[A.Start() + j][column] * tempSend[i];
      }
    }

    MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
    MPI_Wait(&recvRequest, MPI_STATUS_IGNORE);

    float *temp = tempSend;
    tempSend = tempRecv;
    tempRecv = temp;
  }
}

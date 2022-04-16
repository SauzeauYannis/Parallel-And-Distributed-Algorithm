#include <3-2-broadcast/Broadcast.h>
#include <MPI/OPP_MPI.h>

#include <algorithm>
#include <cstring>
#include <iostream>

// Version pipeline ...
void Broadcast(const int k,      // numéro du processeur émetteur, dans 0..P-1
               int *const addr,  // pointeur sur les données à envoyer/recevoir
               const int N,      // nombre d'entiers à envoyer/recevoir
               const int M       // taille d'un paquet de données ...
) {
  OPP::MPI::Ring ring(MPI_COMM_WORLD);

  if (ring.getRank() == k) {
    for (int i = 0; i < N / M; ++i) ring.Send(&addr[i * M], M, MPI_INT);
  } else if (ring.getNext() == k) {
    for (int i = 0; i < N / M; ++i) ring.Recv(&addr[i * M], M, MPI_INT);
  } else {
    ring.Recv(addr, M, MPI_INT);
    for (int i = 0; i < (N / M) - 1; ++i) {
      MPI_Request request = ring.AsyncSend(&addr[i * M], M, MPI_INT);
      ring.Recv(&addr[(i + 1) * M], M, MPI_INT);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    ring.Send(&addr[((N / M) - 1) * M], M, MPI_INT);
  }
}

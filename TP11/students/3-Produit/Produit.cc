#include <3-Produit/Produit.h>
#include <MPI/OPP_MPI.h>

#include <memory>

namespace {}  // namespace

void Produit(const OPP::MPI::Torus &torus, const DistributedBlockMatrix &A,
             const DistributedBlockMatrix &B, DistributedBlockMatrix &C) {
  const double n = sqrt(torus.getCommunicator().size);
  const int i = torus.getRowRing().getRank();
  const int j = torus.getColumnRing().getRank();

  std::cout << "n = " << n << " i = " << i << " j = " << j << std::endl;
}

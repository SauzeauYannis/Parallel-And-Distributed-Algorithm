#include <MPI/OPP_MPI.h>

#include <iostream>

using namespace std;

// entry function
int main(int argc, char **argv) {
  const int level =
      OPP::MPI::Initializer::init(&argc, &argv, MPI_THREAD_MULTIPLE);

  // get the world rank
  OPP::MPI::Communicator communicator(MPI_COMM_WORLD);
  if (communicator.rank == 0)
    std::cout << "Level of multi-threading support by MPI: " << level
              << std::endl;

  // split the world into groups according to logical grid topology
  OPP::MPI::Torus torus = OPP::MPI::Torus::build(communicator.communicator);

  // now, just display the row and column ranks
  std::cout << "[" << communicator.rank
            << "] (x,y) = " << torus.getRowRing().getRank() << "/"
            << torus.getColumnRing().getRank() << std::endl;

  OPP::MPI::Initializer::close();
  return 0;
}

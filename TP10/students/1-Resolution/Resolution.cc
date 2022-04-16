#include <1-Resolution/Resolution.h>
#include <mpi.h>

#include <algorithm>

void Solve(const OPP::MPI::Communicator& communicator,
           const DistributedRowMatrix& L, const DistributedBlockVector& B,
           DistributedBlockVector& X, const int N) {
  OPP::MPI::Ring ring(communicator);

  // Here, we have a block of row (take care to the distribution!)
  // block size ... or B.End() - B.Start() except the last processor (it can be
  // smaller for last block)
  const int m = (N + ring.getSize() - 1) / ring.getSize();
  // check it is ok
  if (m < B.End() - B.Start())
    std::cerr << "Bad value for m=" << m << std::endl;

  float* newB = new float[B.End() - B.Start()];

  for (int i = B.Start(), n = 0; i < B.End(); i++, n++) newB[n] = B[i];

  if (ring.getRank() == 0) {
    for (int col = 0; col < B.End(); ++col) {
      X[col] = newB[col] / L[col][col];

      for (int line = col; line < B.End(); ++line)
        newB[line] -= L[line][col] * X[col];
    }

    ring.Send(&X[0], m, MPI_FLOAT);
  } else {
    float* prevX = new float[m * ring.getRank()];

    for (int proc = 0; proc < ring.getRank(); proc++) {
      ring.Recv(prevX + m * proc, m, MPI_FLOAT);

      if (ring.getNext() != 0) ring.Send(prevX + m * proc, m, MPI_FLOAT);

      for (int col = proc * m; col < ((proc + 1) * m); col++) {
        for (int line = B.Start(); line < B.End(); ++line)
          newB[line - B.Start()] -= L[line][col] * prevX[col];
      }
    }

    for (int col = B.Start(); col < B.End(); col++) {
      X[col] = newB[col - B.Start()] / L[col][col];

      for (int line = col; line < B.End(); ++line)
        newB[line - B.Start()] -= L[line][col] * X[col];
    }

    if (ring.getNext() != 0) ring.Send(&X[m * ring.getRank()], m, MPI_FLOAT);
  }
}

/*
 *  This function is used to print the algorithm steps
 */

/*
void Solve(const OPP::MPI::Communicator& communicator,
           const DistributedRowMatrix& L, const DistributedBlockVector& B,
           DistributedBlockVector& X, const int N) {
  OPP::MPI::Ring ring(communicator);

  // Here, we have a block of row (take care to the distribution!)
  // block size ... or B.End() - B.Start() except the last processor (it can be
  // smaller for last block)
  const int m = (N + ring.getSize() - 1) / ring.getSize();
  // check it is ok
  if (m < B.End() - B.Start())
    std::cerr << "Bad value for m=" << m << std::endl;

  float* newB = new float[B.End() - B.Start()];

  for (int i = B.Start(), n = 0; i < B.End(); i++, n++) newB[n] = B[i];

  std::cout << "Processor " << ring.getRank() << ": " << B.End() - B.Start()
            << " rows" << std::endl;
  std::cout << "Processor " << ring.getRank() << ": m=" << m << std::endl;
  std::cout << "Processor " << ring.getRank() << ": B.Start()=" << B.Start()
            << " B.End()=" << B.End() << std::endl;

  if (ring.getRank() == 0) {
    for (int col = 0; col < B.End(); ++col) {
      X[col] = newB[col] / L[col][col];
      std::cout << "Rank " << ring.getRank() << " X[" << col << "]=" << X[col]
                << std::endl;

      for (int line = col; line < B.End(); ++line) {
        std::cout << "Rank " << ring.getRank() << " prevB[" << line
                  << "]=" << newB[line] << std::endl;
        newB[line] -= L[line][col] * X[col];
        std::cout << "Rank " << ring.getRank() << " newB[" << line
                  << "]=" << newB[line] << std::endl;
      }
    }

    ring.Send(&X[0], m, MPI_FLOAT);
    std::cout << "Rank " << ring.getRank() << " sent " << m << " elements: ";
    for (int i = 0; i < m; i++) std::cout << X[i] << " ";
    std::cout << std::endl;
  }

  else {
    std::cout << "Rank " << ring.getRank()
              << " m * ring.getRank()=" << m * ring.getRank() << std::endl;
    float* prevX = new float[m * ring.getRank()];

    for (int proc = 0; proc < ring.getRank(); proc++) {
      ring.Recv(prevX + m * proc, m, MPI_FLOAT);
      std::cout << "Rank " << ring.getRank() << " received " << m
                << " elements from rank" << proc << ": ";
      for (int i = m * proc; i < m + m * proc; i++)
        std::cout << prevX[i] << " ";
      std::cout << std::endl;

      if (ring.getNext() != 0) {
        ring.Send(prevX + m * proc, m, MPI_FLOAT);
        std::cout << "Rank " << ring.getRank() << " sent " << m
                  << " elements: ";
        for (int i = m * proc; i < m + m * proc; i++)
          std::cout << prevX[i] << " ";
        std::cout << std::endl;
      }

      for (int col = proc * m; col < ((proc + 1) * m); col++) {
        for (int line = B.Start(); line < B.End(); ++line) {
          std::cout << "Rank " << ring.getRank() << " prevB["
                    << line - B.Start() << "]=" << newB[line - B.Start()]
                    << std::endl;
          newB[line - B.Start()] -= L[line][col] * prevX[col];
          std::cout << "Rank " << ring.getRank() << " newB[" << line - B.Start()
                    << "]=" << newB[line - B.Start()] << std::endl;
        }
      }
    }

    for (int col = B.Start(); col < B.End(); col++) {
      X[col] = newB[col - B.Start()] / L[col][col];
      std::cout << "Rank " << ring.getRank() << " X[" << col << "]=" << X[col]
                << std::endl;

      for (int line = col; line < B.End(); ++line) {
        std::cout << "Rank " << ring.getRank() << " prevB[" << line - B.Start()
                  << "]=" << newB[line - B.Start()] << std::endl;
        newB[line - B.Start()] -= L[line][col] * X[col];
        std::cout << "Rank " << ring.getRank() << " newB[" << line - B.Start()
                  << "]=" << newB[line - B.Start()] << std::endl;
      }
    }

    if (ring.getNext() != 0) {
      ring.Send(&X[m * ring.getRank()], m, MPI_FLOAT);
      std::cout << "Rank " << ring.getRank() << " sent " << m << " elements: ";
      for (int i = m * ring.getRank(); i < m + m * ring.getRank(); i++)
        std::cout << X[i] << " ";
      std::cout << std::endl;
    }
  }
}
*/

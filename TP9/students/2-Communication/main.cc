#include <MPI/OPP_MPI.h>

namespace {

int receiveOrInitIfFirstProc(const OPP::MPI::Ring &ring) {
  // first processor?
  if (ring.getRank() == 0) {
    std::cout << "Processor " << ring.getRank() << " init data" << std::endl;
    // yep, so return the answer
    return 0b101010;
  }

  // no, so receive from previous one
  int number = -1;
  ring.Recv(&number, 1, MPI_INT);

  std::cout << "Processor " << ring.getRank() << " receive" << std::endl;

  return number;
}

void sendToNextProcIfNotLast(const OPP::MPI::Ring &ring, int number) {
  // if last processor, just do nothing
  if (ring.getNext() == 0) return;
  // not last, so send to next processor
  ring.Send(&number, 1, MPI_INT);

  std::cout << "Processor " << ring.getRank() << " send " << number
            << " to the next" << std::endl;
}

int test(const OPP::MPI::Communicator &comm) {
  if (comm.size < 2) {
    std::cerr << "You must launch at least two processors ... aborting"
              << std::endl;
    return -1;
  }

  OPP::MPI::Ring ring(comm.communicator);

  sendToNextProcIfNotLast(ring, receiveOrInitIfFirstProc(ring));

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  const int err_code = test(MPI_COMM_WORLD);

  MPI_Finalize();
  
  return err_code;
}
#include <exo2/student.h>

#include <algorithm>
#include <functional>
#include <iostream>

namespace {

inline unsigned extract(unsigned v, unsigned bit) {
  if (bit == 0) return 1 - (v & 0x1u);
  return 1 - ((v >> bit) & 0x1u);
}

void run_partition_sequential(std::vector<unsigned>& input,
                              std::vector<unsigned>& predicate,
                              std::vector<unsigned>& output) {
  for (int i = 1, n = 0; i >= 0; --i) {
    for (size_t j = 0; j < input.size(); ++j) {
      if (predicate[j] == i) output[n++] = input[j];
    }
  }
}

}  // namespace

bool StudentWorkImpl::isImplemented() const { return true; }

void StudentWorkImpl::run_radixSort_sequential(std::vector<unsigned>& input,
                                               std::vector<unsigned>& output) {
  using wrapper = std::reference_wrapper<std::vector<unsigned>>;

  std::vector<unsigned> temp(input.size());

  wrapper T[2] = {wrapper(output), wrapper(temp)};

  std::vector<unsigned> predicate(input.size());

  std::copy(input.begin(), input.end(), output.begin());

  for (unsigned numeroBit = 0; numeroBit < sizeof(unsigned) * 8; ++numeroBit) {
    const int ping = numeroBit & 1;
    const int pong = 1 - ping;

    for (size_t i = 0; i < input.size(); ++i)
      predicate[i] = extract(T[ping].get()[i], numeroBit);

    run_partition_sequential(T[ping], predicate, T[pong]);
  }
}

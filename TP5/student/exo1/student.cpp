#include <exo1/student.h>

#include <iostream>

namespace {}

bool StudentWorkImpl::isImplemented() const { return true; }

void StudentWorkImpl::run_partition_sequential(std::vector<int>& input,
                                               std::vector<int>& predicate,
                                               std::vector<int>& output) {
  for (int i = 1, n = 0; i >= 0; --i) {
    for (size_t j = 0; j < input.size(); ++j) {
      if (predicate[j] == i) output[n++] = input[j];
    }
  }
}

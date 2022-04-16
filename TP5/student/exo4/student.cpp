#include <exo4/partition.h>
#include <exo4/student.h>

#include <iostream>

namespace {}

bool StudentWorkImpl::isImplemented() const { return true; }

void StudentWorkImpl::run_partition_parallel(std::vector<int>& input,
                                             std::vector<int>& predicate,
                                             std::vector<int>& output) {
  OPP::partition(input.begin(), input.end(), predicate.begin(), output.begin());
}
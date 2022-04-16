#pragma once

#include <StudentWork.h>

#include <vector>

class StudentWorkImpl : public StudentWork {
 public:
  bool isImplemented() const;

  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl&) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

  void run_partition_parallel(std::vector<int>& input,
                              std::vector<int>& predicate,
                              std::vector<int>& output);
};
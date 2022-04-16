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

  void run_radixSort_sequential(std::vector<unsigned>& input,
                                std::vector<unsigned>& output);
};
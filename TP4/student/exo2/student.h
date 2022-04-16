#pragma once

#include <StudentWork.h>

#include <functional>
#include <vector>

#include "exclusive_scan.h"

class StudentWorkImpl : public StudentWork {
 public:
  bool isImplemented() const { return true; }

  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl&) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

  template <typename T>
  void run_scan_sequential(std::vector<T>& input, std::vector<T>& output,
                           const T& Tinit, std::function<T(T, T)>& functor) {
    OPP::exclusive_scan_seq(input.begin(), input.end(), output.begin(),
                            std::forward<std::function<T(T, T)>>(functor),
                            Tinit);
  }

  template <typename T>
  void run_scan_parallel(std::vector<T>& input, std::vector<T>& output,
                         const T& Tinit, std::function<T(T, T)>& functor) {
    OPP::exclusive_scan_par(input.begin(), input.end(), output.begin(),
                            std::forward<std::function<T(T, T)>>(functor),
                            Tinit);
  }
};
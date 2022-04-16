#pragma once

#include <StudentWork.h>
#include <exo4/partition.h>
#include <previous/transform.h>

#include <functional>
#include <vector>

class StudentWorkImpl : public StudentWork {
 public:
  bool isImplemented() const;

  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl&) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

  template <typename T>
  inline unsigned extract(const T& v, const unsigned& bit) {
    if (bit) return 1 - ((v >> bit) & 0x1u);
    return 1 - (v & 0x1u);
  }

  template <typename T>
  void run_radixSort_parallel(std::vector<T>& input, std::vector<T>& output) {
    std::copy(input.begin(), input.end(), output.begin());
    std::vector<T> temp(input.size());
    std::vector<T>* array[2] = {&output,
                                &temp};  // des pointeurs conviennent aussi !

    std::vector<unsigned> predicate(input.size());
    std::copy(input.begin(), input.end(), output.begin());
    
    for (unsigned numeroBit = 0; numeroBit < sizeof(T) * 8u; ++numeroBit) {
      const int ping = numeroBit & 1;
      const int pong = 1 - ping;

      OPP::transform(array[ping]->begin(), array[ping]->end(),
                     predicate.begin(),
                     std::function([this, &numeroBit](const T& value) -> T {
                       return extract(value, numeroBit);
                     }));

      OPP::partition(array[ping]->begin(), array[ping]->end(),
                     predicate.begin(), array[pong]->begin());
    }
  }
};
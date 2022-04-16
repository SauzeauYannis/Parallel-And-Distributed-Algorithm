#pragma once

#include <OPP.h>
#include <StudentWork.h>
#include <previous/exclusive_scan.h>
#include <previous/inclusive_scan.h>
#include <previous/scatter.h>
#include <previous/transform.h>

#include <iostream>
#include <vector>

class StudentWorkImpl : public StudentWork {
 public:
  bool isImplemented() const;

  StudentWorkImpl() = default;
  StudentWorkImpl(const StudentWorkImpl&) = default;
  ~StudentWorkImpl() = default;
  StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

  template <typename T>
  inline T extract(const T& v, const T& bit) {
    if (bit == 0) return 1 - (v & 0x1u);
    return 1 - ((v >> bit) & 0x1u);
  }

  template <typename T>
  void run_radixSort_parallel(const std::vector<T>& input,
                              std::vector<T>& output) {
    const unsigned n = input.size();

    using wrapper = std::reference_wrapper<std::vector<T>>;
    std::vector<T> temp(n);
    wrapper W[2] = {wrapper(output), wrapper(temp)};
    std::copy(input.begin(), input.end(), output.begin());

    std::vector<unsigned> iDown(n);
    std::vector<unsigned> iUp(n);

    for (T numeroBit = 0; numeroBit < sizeof(T) * 8; ++numeroBit) {
      const int ping = numeroBit & 1;
      const int pong = 1 - ping;

      const std::vector<T>& in = W[ping].get();

      const OPP::TransformIterator predicate = OPP::make_transform_iterator(
          in.begin(), std::function([this, &numeroBit](const T& v) -> T {
            return extract(v, numeroBit);
          }));

      OPP::exclusive_scan_par(predicate + 0, predicate + n, iDown.begin(),
                              std::plus<T>(), T(0));

      const OPP::TransformIterator not_predicate_reversed = OPP::make_transform_iterator(
              OPP::CountingIterator(1l),
              std::function([&predicate, &n](const T& a) -> T {
                return 1 - predicate[n - a];
              }));

      OPP::inclusive_scan_par(not_predicate_reversed + 0,
                              not_predicate_reversed + n, iUp.rbegin(),
                              std::plus<T>());

      OPP::scatter(
          in.begin(), in.end(),
          OPP::make_transform_iterator(
              OPP::CountingIterator(0l),
              std::function([&predicate, &iDown, &iUp, &n](const T& a) -> T {
                if (predicate[a]) return iDown[a];
                return n - iUp[a];
              })),
          W[pong].get().begin());
    }
  }

  void check();
};
#pragma once

#include <OPP.h>

#include <algorithm>
#include <future>
#include <ranges>
#include <thread>
#include <vector>

namespace OPP {
template <typename InputIteratorType, typename OutputIteratorType,
          typename MapFunction>
inline void transform(const InputIteratorType&& aBegin,  // left operand
                      const InputIteratorType&& aEnd,
                      OutputIteratorType&& oBegin,  // destination
                      const MapFunction&& functor   // unary functor
) {
  int nbTasks = OPP::nbThreads;

  int fullSize = aEnd - aBegin;
  int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

  if (fullSize < nbTasks) {
    for (auto iter = aBegin; iter < aEnd; ++iter)
      oBegin[iter - aBegin] = functor(*iter);
    return;
  }

  std::vector<std::shared_future<void>> futures;

  for (int i = 0; i < nbTasks; ++i) {
    int start = i * chunkSize;
    int end = std::min(start + chunkSize, fullSize);

    if (start >= end) break;

    futures.emplace_back(std::move(OPP::getDefaultThreadPool().push_task(
        [start, end, aBegin, oBegin, functor](void) -> void {
          for (int iter = start; iter < end; ++iter)
            oBegin[iter] = functor(aBegin[iter]);
        })));
  }

  for (auto&& future : futures) future.get();
}

// second version: two input iterators!
template <typename InputIteratorType, typename OutputIteratorType,
          typename MapFunction>
inline void transform(const InputIteratorType&& aBegin,  // left operand
                      const InputIteratorType&& aEnd,
                      const InputIteratorType&& bBegin,  // right operand
                      OutputIteratorType&& oBegin,       // destination
                      const MapFunction&& functor        // binary functor
) {
  int nbTasks = OPP::nbThreads;

  int fullSize = aEnd - aBegin;
  int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

  if (fullSize < nbTasks) {
    for (auto iter = aBegin; iter < aEnd; ++iter)
      oBegin[iter - aBegin] = functor(*iter, bBegin[iter - aBegin]);
    return;
  }

  std::vector<std::shared_future<void>> futures;

  for (int i = 0; i < nbTasks; ++i) {
    int start = i * chunkSize;
    int end = std::min(start + chunkSize, fullSize);

    if (start >= end) break;

    futures.emplace_back(std::move(OPP::getDefaultThreadPool().push_task(
        [start, end, aBegin, bBegin, oBegin, functor](void) -> void {
          for (int iter = start; iter < end; ++iter)
            oBegin[iter] = functor(aBegin[iter], bBegin[iter]);
        })));
  }

  for (auto&& future : futures) future.get();
}

};  // namespace OPP

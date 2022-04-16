#pragma once
#include <OPP.h>

#include <algorithm>
#include <future>
#include <ranges>
#include <thread>
#include <vector>

// scatter is a permutation of data. The destination index is given thanks to an
// iterator.

namespace OPP {

template <typename InputIteratorType, typename MapIteratorType,
          typename OutputIteratorType>
inline void scatter(const InputIteratorType&& aBegin,  // left operand
                    const InputIteratorType&& aEnd,
                    const MapIteratorType&& map,  // source index
                    OutputIteratorType&& oBegin   // destination
) {
  int nbTasks = OPP::nbThreads;

  int fullSize = aEnd - aBegin;
  int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

  if (fullSize < nbTasks) {
    for (auto iter = aBegin; iter < aEnd; ++iter)
      oBegin[map[iter - aBegin]] = aBegin[iter - aBegin];
    return;
  }

  std::vector<std::shared_future<void>> futures;

  for (int i = 0; i < nbTasks; ++i) {
    int start = i * chunkSize;
    int end = std::min(start + chunkSize, fullSize);

    if (start >= end) break;

    futures.emplace_back(std::move(OPP::getDefaultThreadPool().push_task(
        [start, end, aBegin, map, oBegin](void) -> void {
          for (int iter = start; iter < end; ++iter)
            oBegin[map[iter]] = aBegin[iter];
        })));
  }

  for (auto&& future : futures) future.get();
}

};  // namespace OPP

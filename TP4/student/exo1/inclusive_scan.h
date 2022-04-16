#pragma once
#include <OPP.h>

#include <algorithm>
#include <iterator>
#include <thread>
#include <vector>

// inclusive scan

namespace OPP {

  template <typename InputIteratorType, typename OutputIteratorType,
            typename BinaryFunction>
  inline void inclusive_scan_seq(
      const InputIteratorType &&aBegin,   // input begin
      const InputIteratorType &&aEnd,     // input end (excluded)
      const OutputIteratorType &&oBegin,  // output begin
      const BinaryFunction &&functor      // should be associative
  ) {
    const int fullSize = aEnd - aBegin;

    oBegin[0] = aBegin[0];
    for (int i = 1; i < fullSize; ++i)
      oBegin[i] = functor(oBegin[i - 1], aBegin[i]);
  }

  template <typename InputIteratorType, typename OutputIteratorType,
            typename BinaryFunction>
  inline void inclusive_scan_par(
      const InputIteratorType &&aBegin,   // input begin
      const InputIteratorType &&aEnd,     // input end (excluded)
      const OutputIteratorType &&oBegin,  // output begin
      const BinaryFunction &&functor      // should be associative
  ) {
    const int nbTasks = OPP::nbThreads * 4;

    const int fullSize = aEnd - aBegin;

    if (fullSize < nbTasks) {
      inclusive_scan_seq(std::move(aBegin), std::move(aEnd), std::move(oBegin),
                        std::move(functor));
      return;
    }

    OPP::ThreadPool &pool = OPP::getDefaultThreadPool();

    std::vector<std::shared_future<void>> futures;

    const int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

    for (int i = 0; i < nbTasks; ++i) {
      const int start = i * chunkSize;
      const int last = std::min(start + chunkSize, fullSize);

      if (start >= last) break;

      futures.emplace_back(
          std::move(pool.push_task([start, last, aBegin, oBegin, functor]() {
            inclusive_scan_seq(aBegin + start, aBegin + last, oBegin + start,
                              std::move(functor));
          })));
    }

    for (auto &&future : futures) future.get();

    std::vector<typename InputIteratorType::value_type> aux(nbTasks - 1);

    aux[0] = oBegin[chunkSize - 1];

    for (int i = 1; i < nbTasks - 1; ++i)
      aux[i] = functor(aux[i - 1], oBegin[chunkSize * (i + 1) - 1]);

    futures.clear();

    for (int i = 0; i < nbTasks - 1; ++i) {
      const int start = (i + 1) * chunkSize;

      futures.emplace_back(std::move(
          pool.push_task([i, start, chunkSize, fullSize, aux, oBegin, functor]() {
            for (int j = 0; j < chunkSize; ++j) {
              if (start + j < fullSize)
                oBegin[start + j] = functor(aux[i], oBegin[start + j]);
            }
          })));
    }

    for (auto &&future : futures) future.get();
  }

};  // namespace OPP
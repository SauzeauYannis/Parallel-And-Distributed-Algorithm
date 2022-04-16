#pragma once
#include <OPP.h>

#include <algorithm>
#include <thread>
#include <vector>

// inclusive scan

namespace OPP {

template <typename InputIteratorType, typename OutputIteratorType,
          typename BinaryFunction,
          typename T>
inline void exclusive_scan_seq(
    const InputIteratorType&& aBegin,   // input begin
    const InputIteratorType&& aEnd,     // input end (excluded)
    const OutputIteratorType&& oBegin,  // output begin
    const BinaryFunction&& functor,     // should be associative
    const T Tinit = T(0)) {
  const int fullSize = aEnd - aBegin;

  oBegin[0] = Tinit;
  for (int i = 1; i < fullSize; ++i)
    oBegin[i] = functor(oBegin[i - 1], aBegin[i - 1]);
}

template <typename InputIteratorType, typename OutputIteratorType,
          typename BinaryFunction,
          typename T>
inline void exclusive_scan_par(
    const InputIteratorType&& aBegin,   // input begin
    const InputIteratorType&& aEnd,     // input end (excluded)
    const OutputIteratorType&& oBegin,  // output begin
    const BinaryFunction&& functor,     // should be associative
    const T Tinit = T(0)) {
  const int nbTasks = OPP::nbThreads;

  const int fullSize = aEnd - aBegin;

  if (fullSize < nbTasks) {
    exclusive_scan_seq(std::move(aBegin), std::move(aEnd), std::move(oBegin),
                       std::move(functor), Tinit);
    return;
  }

  OPP::ThreadPool& pool = OPP::getDefaultThreadPool();

  std::vector<std::shared_future<void>> futures;

  const int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

  for (int i = 0; i < nbTasks; ++i) {
    const int start = i * chunkSize;
    const int last = std::min(start + chunkSize, fullSize);

    if (start >= last) break;

    futures.emplace_back(std::move(
        pool.push_task([start, last, Tinit, aBegin, oBegin, functor]() {
          exclusive_scan_seq(aBegin + start, aBegin + last, oBegin + start,
                             std::move(functor), Tinit);
        })));
  }

  for (auto&& future : futures) future.get();

  std::vector<T> aux(nbTasks - 1);

  aux[0] = functor(oBegin[chunkSize - 1], aBegin[chunkSize - 1]);
  for (int i = 1; i < nbTasks - 1; ++i)
    aux[i] = functor(aux[i - 1], functor(oBegin[chunkSize * (i + 1) - 1],
                                         aBegin[chunkSize * (i + 1) - 1]));

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

  for (auto&& future : futures) future.get();
}

};  // namespace OPP
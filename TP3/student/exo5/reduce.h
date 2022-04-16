#pragma once
#include <OPP.h>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <ranges>

// gather is a permutation of data. The source index is given thanks to an iterator. 

namespace OPP {
    
    template<   typename InputIteratorType, 
                typename T, 
                typename MapFunction>
        inline
    T reduce(
        const InputIteratorType&& aBegin, 
        const InputIteratorType&& aEnd,
        const T&& init,
        const MapFunction&& functor // unary functor
    ) {
        int nbTasks = OPP::nbThreads * 4;

        int fullSize = aEnd - aBegin;
        int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

        if (fullSize < nbTasks) {
            T sum = init;
             for (auto iter = aBegin; iter < aEnd; ++iter)
                sum = functor(sum, aBegin[iter - aBegin]);
            return sum;
        }

        std::vector<std::shared_future<T>> futures;
        
        for (int i = 0; i < nbTasks; ++i) {
            int start = i * chunkSize;
            int end = std::min(start + chunkSize, fullSize);

            if (start >= end) break;

            futures.emplace_back(
                std::move(OPP::getDefaultThreadPool().push_task(
                    [start, end, aBegin, functor] (void) -> T {
                        T acc = T(0);
                        for (int iter = start; iter < end; ++iter)
                            acc = functor(acc, aBegin[iter]);
                        return acc;
                    }
                ))
            );
        }

        T sum = init;

        for (auto &&future : futures)
            sum += future.get();

        return sum;
    }
};
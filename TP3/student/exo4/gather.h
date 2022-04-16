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
                typename MapIteratorType, 
                typename OutputIteratorType>
        inline
    void gather(
        const InputIteratorType&& aBegin, // left operand
        const InputIteratorType&& aEnd,
        const MapIteratorType&& map, // source index
        OutputIteratorType&& oBegin // destination
    ) {
        int nbTasks = OPP::nbThreads * 4;

        int fullSize = aEnd - aBegin;
        int chunkSize = (fullSize + nbTasks - 1) / nbTasks;

        if (fullSize < nbTasks) {
            for (auto iter = aBegin; iter < aEnd; ++iter)
                oBegin[iter - aBegin] = map[iter - aBegin];
            return;
        }

        std::vector<std::shared_future<void>> futures;
        
        for (int i = 0; i < nbTasks; ++i) {
            int start = i * chunkSize;
            int end = std::min(start + chunkSize, fullSize);

            if (start >= end) break;

            futures.emplace_back(
                std::move(OPP::getDefaultThreadPool().push_task(
                    [start, end, aBegin, map, oBegin] (void) -> void { 
                        for (int iter = start; iter < end; ++iter)
                            oBegin[iter] = map[aBegin[iter]];
                    }
                ))
            );
        }
        /*
        for (int i = 0; i < nbTasks; ++i) {
            futures.emplace_back(
                std::move(OPP::getDefaultThreadPool().push_task(
                    [i, nbTasks, fullSize, aBegin, map, oBegin] (void) -> void { 
                        for (int iter = i; iter < fullSize; iter += nbTasks)
                            oBegin[iter] = map[aBegin[iter]];
                    }
                ))
            );
        }
        */
        for (auto &&future : futures)
            future.get();
    }

};

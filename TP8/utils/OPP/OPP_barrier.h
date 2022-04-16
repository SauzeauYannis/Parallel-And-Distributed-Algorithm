#pragma once
#include <thread>
#include <condition_variable>
#include <OPP.h>

namespace OPP
{
    /** 
     * This class implements a thread-barrier. 
     * The constructor needs the number of used threads.
     * When a thread calls the "arrive_and_wait()" method, 
     * it will sleep until the number of sleeping threads
     * equals the number of used threads. Then, they will 
     * be wake up ...
     * The barrier may be used once, twice, and so on.
     */
    class Barrier 
    {
        std::mutex mutex;
        std::condition_variable cv;
        const unsigned nbThreadsInBarrier;
        unsigned nbExpectedThreads;

    public:
        Barrier(const unsigned nb) 
            : nbThreadsInBarrier(nb), nbExpectedThreads(nbThreads) 
        {}
        
        ~Barrier() = default;
        Barrier(const Barrier&) = delete;
        Barrier& operator=(const Barrier&) = delete;

        void arrive_and_wait() {
            std::unique_lock<std::mutex> lock(mutex);
            -- nbExpectedThreads;
            if( nbExpectedThreads > 0 )
                cv.wait(lock);
            else {
                nbExpectedThreads = nbThreadsInBarrier; // reset
                cv.notify_all();
            }
            lock.unlock();
        }
    };

}
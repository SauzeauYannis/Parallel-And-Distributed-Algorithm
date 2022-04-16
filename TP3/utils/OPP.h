#pragma warning(disable : 4996)
#pragma once
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#include <thread>
#include <condition_variable>
#include <iterator>
#include <functional>
#include <vector>
#include <queue>
#include <mutex>
#include <future>
#include <algorithm>
#include <optional>


// we define Our Parallel Pattern namespace -> OPP
namespace OPP 
{
    extern unsigned nbThreads;

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
                cv.wait(lock, [=]() {
                    return nbExpectedThreads == nbThreadsInBarrier;
                });
            else {
                nbExpectedThreads = nbThreadsInBarrier; // reset
                cv.notify_all();
            }
        }
    };

    /** 
     * This class implements a thread semaphore.
     * 
     * Actually, counting and binary semaphore are available in C++ ... from version 20!
     * 
     * The constructor is also the initialiser of the counting value.
     */    
    template<typename T=uint32_t>
    class Semaphore 
    {
        std::mutex mutex;
        std::condition_variable cv;
        T value;

    public:
        // creates (and initializes) a new Semaphore
        Semaphore(const T init=T(0)) 
            : value(init)
        {}
        
        ~Semaphore() = default;
        Semaphore(const Semaphore&) = delete;
        Semaphore& operator=(const Semaphore&) = delete;        

        // Acquires the semaphore (P)
        void acquire(const T how=T(1)) 
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [=] { return value >= std::max(T(1),how) ;});
            value -= std::max(T(1),how);
        }
        
        // Releases the semaphore (V)
        void release(const T how=T(1)) 
        {
            const std::lock_guard<std::mutex> lock(mutex);
            T n = std::max(T(1),how);
            value += n; // warning, here an overflow may occur ...
            while(n--) 
                cv.notify_one();
        }
    };

    class ThreadPool
    {       
        // need some polymorphism to be able to return a correct std::shared_future ...
        struct Job 
        {
            virtual ~Job() = default;
            virtual void run() = 0;
        };
        // the use class to store real tasks
        template <typename R>
        struct AnyJob : public Job 
        {
            std::packaged_task<R()> task;
            AnyJob(std::packaged_task<R()> func) : task(std::move(func)) {}
            void run() 
            {
                task();
            }
        }; 

        // queue of jobs
        class jobs_queue 
        {
            using ulock = std::unique_lock<std::mutex>;
            using glock = std::lock_guard<std::mutex>;

        public:    
            void push(std::shared_ptr<Job> job) 
            {
                glock l(mutex);
                queue.emplace(std::move(job));
                condition_variable.notify_one();
            }
            
            std::optional<std::shared_ptr<Job>> pop() 
            {
                ulock l(mutex);
                condition_variable.wait(l, [this]{ return abort || !queue.empty(); } );
                if (abort) 
                    return std::nullopt;
                auto job = queue.front();
                queue.pop();
                return job;
            }

            void terminate() 
            {
                glock l(mutex);
                abort = true;
                condition_variable.notify_all();
            }

            ~jobs_queue()
            {
                terminate();
            }

        private:
            std::mutex mutex;
            std::queue<std::shared_ptr<Job>> queue;
            std::condition_variable condition_variable;
            bool abort = false;
        }; // class job_queues

        const uint32_t maxThreads = uint32_t(std::thread::hardware_concurrency());

        // list of workers (threads)
        std::vector<std::thread> workers;

        // our threads' pool contains a list of jobs to run
        jobs_queue jobs;
    
        // add workers, setting the function they will run 
        void add_workers(const uint32_t n = 1) 
        {
            for(uint32_t i=0; i<n; ++i) 
            {
                // creates one thread with infinite loop
                std::thread thread( 
                    [=] {
                        while (true) 
                        {
                            // get one job (blocking function)
                            auto optional_job = jobs.pop();
                            // check if exit is requested
                            if (!optional_job)
                                break; // exit requested
                            // get the job from the optional value
                            std::shared_ptr<Job> job = (*optional_job);
                            // and run the job !
                            job->run();
                        }
                    }
                );
                // put the thread into the vector of workers
                workers.emplace_back( std::move(thread) );
            }
        }

    public:

        // push a R() function, i.e. function returning R and taking no arguments
        // NB : void() functions are allowed ;-)
        template <typename _Fct>
        auto push_task( _Fct&& f ) 
        {
            // gets the returning type
            using R = typename std::invoke_result<_Fct>::type; // Warning: since C++17
            // make a task from the  function, allowing to get at "future"
            std::packaged_task<R()> task(std::move(f));
            // gets the future
            std::shared_future<R> future = task.get_future();
            // push the task into the jobs queue
            jobs.push( std::make_shared<AnyJob<R>>( std::move(task) ) ); 
            // returns the future
            return future;
        }

        // returns the real number of threads ...
        auto getRealThreadNumber()
        {
            return workers.size();
        }

        // constructor 
        ThreadPool(uint32_t nbThreads)
        {
            add_workers(std::min(nbThreads, maxThreads));
        }

        ThreadPool() = delete;
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool &operator=(const ThreadPool&) = delete;

        // do some cleaning ... stop the threads!
        ~ThreadPool() 
        {
            jobs.terminate();
            std::for_each(workers.begin(), workers.end(), [] (std::thread &th) { th.join();});
        }
    };

    ThreadPool& getDefaultThreadPool(); 

    // counting iterator ... T should be an integer (char/short/int/long/long long, signed or unsigned)
    template<
        typename T=long,
        typename Tdiff=long long>
    class CountingIterator : public std::iterator<std::random_access_iterator_tag, T>
    {
        T position;
    public:
        using pointer = typename std::iterator<std::random_access_iterator_tag,T>::pointer;
        using reference = typename std::iterator<std::random_access_iterator_tag,T>::reference;
        
        CountingIterator(const T position=T(0)) 
            : position(position) 
        {}
        CountingIterator(const CountingIterator& cit) = default;
        ~CountingIterator() = default;
        CountingIterator& operator=(const CountingIterator&) = default;
        CountingIterator& operator++() { 
            ++position;
            return *this;
        }
        CountingIterator operator++(int) const {
            return CountingIterator(position++);
        }
        CountingIterator& operator--() { 
            --position;
            return *this;
        }
        CountingIterator operator--(int) const {
            return CountingIterator(position--);
        }
        bool operator==(const CountingIterator& cit) const {
            return position == cit.position;
        }
        bool operator!=(const CountingIterator& cit) const {
            return position != cit.position ;
        }
        T operator*() const { 
            return position; 
        }
        T& operator*() { 
            return position; 
        }
        CountingIterator operator+(const Tdiff& dt) const {
            return CountingIterator(T(position+dt));
        }
        CountingIterator& operator+=(const Tdiff& dt) {
            position += dt;
            return *this;
        }
        CountingIterator operator-(const Tdiff& dt) const {
            return CountingIterator(T(position-dt));
        }
        CountingIterator& operator-=(const Tdiff& dt) {
            position -= dt;
            return *this;
        }        
        T operator[](const Tdiff& n) const {
            return position+n;
        }
        bool operator<(const CountingIterator& cit) const {
            return position < cit.position;
        }

        bool operator>(const CountingIterator& cit) const {
            return position > cit.position;
        }

        bool operator<=(const CountingIterator& cit) const {
            return position <= cit.position;
        }

        bool operator>=(const CountingIterator& cit) const {
            return position >= cit.position;
        }

        Tdiff operator+(const CountingIterator& cit) const {
            return position + cit.position;
        }

        Tdiff operator-(const CountingIterator& cit) const {
            return position - cit.position;
        }
    };

    template<typename T>
    CountingIterator<T> make_counting_iterator(T position) {
        return CountingIterator(position);
    }

    template<
        typename Iterator,
        typename Functor,
        typename Tsrc, 
        typename Tdst,
        typename Tdiff=long long
    >
    class TransformIterator : public std::iterator<std::random_access_iterator_tag, Tdst, Tdiff,void,void>
    {
        Functor  transform;
        Iterator iterator;

    public:
        TransformIterator(const Iterator& iterator, const Functor& transform) 
            : transform(transform), iterator(iterator) 
        {}
        TransformIterator(Iterator&& iterator, Functor&& transform) 
            : transform(transform), iterator(iterator) 
        {}
        TransformIterator(const TransformIterator&) = default;
        ~TransformIterator() = default;
        TransformIterator& operator=(const TransformIterator&) = default;
                
        TransformIterator& operator++() { 
            ++iterator;
            return *this;
        }
        TransformIterator operator++(int) const {
            auto copy = TransformIterator(iterator, transform);
            ++iterator;
            return copy;
        }
        TransformIterator& operator--() { 
            --iterator;
            return *this;
        }
        TransformIterator operator--(int) const {
            auto copy = TransformIterator(iterator, transform);
            --iterator;
            return copy;
        }
        bool operator==(const TransformIterator& cit) const {
            return iterator == cit.iterator && transform == cit.transform;
        }
        bool operator!=(const TransformIterator& cit) const {
            return ! ( *this == cit );
        }
        Tdst operator*() const { 
            return std::invoke(transform, *iterator); 
        }
        TransformIterator operator+(const Tdiff& dt) const {
            return TransformIterator(iterator+dt, transform);
        }
        TransformIterator& operator+=(const Tdiff& dt) {
            iterator += dt;
            return *this;
        }
        TransformIterator operator-(const Tdiff& dt) const {
            return TransformIterator(iterator-dt, transform);
        }
        TransformIterator& operator-=(const Tdiff& dt) {
            iterator -= dt;
            return *this;
        }        
        Tdst operator[](const Tdiff& n) const {
            return std::invoke(transform, iterator[n]);
        }
        bool operator<(const TransformIterator& cit) const {
            return iterator < cit.iterator;
        }

        bool operator>(const TransformIterator& cit) const {
            return iterator > cit.iterator;
        }

        bool operator<=(const TransformIterator& cit) const {
            return iterator <= cit.iterator;
        }

        bool operator>=(const TransformIterator& cit) const {
            return iterator >= cit.iterator;
        }

        Tdiff operator+(const TransformIterator& cit) const {
            return iterator + cit.iterator;
        }

        Tdiff operator-(const TransformIterator& cit) const {
            return iterator - cit.iterator;
        }
    };
    
    template<
        typename Iterator,
        typename Tsrc,
        typename Tdst,
        typename Tdiff=long long>
    inline
    auto make_transform_iterator(Iterator iterator, std::function<Tdst(Tsrc)> functor) 
    {
        //using Tsrc = typename Iterator::value_type;
        //using Tdst = typename functor::result_type; // n�cessite une std::function<Tdst(Tsrc)> ....
        return TransformIterator<Iterator,std::function<Tdst(Tsrc)>,Tsrc,Tdst,Tdiff>(iterator, functor);
    }
}
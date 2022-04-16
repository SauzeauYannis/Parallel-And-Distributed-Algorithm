#include <OPP.h>

using namespace OPP;

ThreadPool& OPP::getDefaultThreadPool()
{
    static ThreadPool* pool = nullptr;
    if( pool == nullptr )
    {
        pool = new ThreadPool(std::thread::hardware_concurrency());
    }
    return *pool;
}

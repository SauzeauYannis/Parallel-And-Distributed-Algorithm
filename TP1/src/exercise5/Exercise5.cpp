#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise5/Exercise5.h>

namespace {
    bool is_prime(const unsigned n) {
        // check division from 2 to n (not efficient at all!)
        for(unsigned d = 2; d<n; ++d)
            if( (n%d) == 0 ) // d is a divisor, n is not prime
                return false;
        // we have not found any divisor: n is prime
        return true;
    }
}

// ==========================================================================================
void Exercise5::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << "[ -t=threads ] [-s=start -e=end]" << std::endl 
        << "where -t  specifies the number of threads (into [1...64]),"
        << "\tand -s and -e specify the interval to consider for twin primes (considering end>=start+2)."
        << std::endl;
}

// ==========================================================================================
void Exercise5::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise5::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}


// ==========================================================================================
Exercise5& Exercise5::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    if( checkCmdLineFlag(argc, argv, "t") ) {
        unsigned value = getCmdLineArgumentInt(argc, argv, "t");
        if( value >  0 && value <= 64 )
            number_of_threads = value;
        else
            usageAndExit(argv[0], -1);  
    }
    bool start_given = checkCmdLineFlag(argc, argv, "s");
    bool end_given = checkCmdLineFlag(argc, argv, "e");
    if( start_given && end_given ) {
        unsigned start = getCmdLineArgumentInt(argc, argv, "s");
        if( start >  0 && start <= (1<<31) )
            interval_start = start;
        else
            usageAndExit(argv[0], -1);  
        unsigned end = getCmdLineArgumentInt(argc, argv, "e");
        if( end >=  start+2 && end <= (1<<31) )
            interval_end = end;
        else
            usageAndExit(argv[0], -1);  
    }
    else if ( start_given || end_given )    
        usageAndExit(argv[0], -1);  
     return *this;
}

// ==========================================================================================
void Exercise5::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Test the student work" << std::endl;
    ChronoCPU chr;
    chr.start();
    twin_primes = reinterpret_cast<StudentWork5*>(student)->run(interval_start, interval_end, number_of_threads);
    chr.stop();
    if( verbose ) 
    {
        std::cout << "\tStudent's Work Done in " << chr.elapsedTimeInMilliSeconds() << " ms" << std::endl;
        for(auto i=twin_primes.begin(); i<twin_primes.end(); i++) 
        {
            std::cout << "-> (" << std::get<0>(*i) << "," << std::get<1>(*i) << ")" << std::endl;
        }
    }
}


// ==========================================================================================
bool Exercise5::check() {
    auto iter_twins=twin_primes.begin();
    for(unsigned i=interval_start; i<=interval_end; ++i) {
        if(is_prime(i) && is_prime(i+2)) 
        { 
            if(i!=std::get<0>(*iter_twins) && i+1!=std::get<1>(*iter_twins))
                return false;
            iter_twins++;
        }
    }
    return iter_twins>=twin_primes.end();;
}


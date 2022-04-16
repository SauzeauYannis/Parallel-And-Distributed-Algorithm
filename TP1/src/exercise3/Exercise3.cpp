#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <exercise3/Exercise3.h>

namespace {
}

// ==========================================================================================
void Exercise3::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << "[ -t=threads ]" << std::endl 
        << "where -t  specifies the number of threads (into [1...64])."
        << std::endl;
}

// ==========================================================================================
void Exercise3::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise3::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise3& Exercise3::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    if( checkCmdLineFlag(argc, argv, "t") ) {
        unsigned value = getCmdLineArgumentInt(argc, argv, "t");
        if( value >  0 && value <= 64 )
            number_of_threads = value;
        else
            usageAndExit(argv[0], -1);  
    }
    return *this;
}

void Exercise3::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Test the student work" << std::endl;
    ChronoCPU chr;
    chr.start();
    pi_student = reinterpret_cast<StudentWork3*>(student)->run(number_of_threads);
    chr.stop();
    if( verbose )
        std::cout << "\tStudent's Work Done in " << chr.elapsedTimeInMilliSeconds() << " ms" << std::endl;
}


bool Exercise3::check() {
    return ( abs(pi_student - M_PI) < 1e-5 );
}


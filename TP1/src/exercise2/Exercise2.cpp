#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <exercise2/Exercise2.h>

namespace {
}

// ==========================================================================================
void Exercise2::usage( const char*const prg ) {
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
void Exercise2::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise2::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise2& Exercise2::parseCommandLine(const int argc, const char**argv) 
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

void Exercise2::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Test the student work" << std::endl;
    ChronoCPU chr;
    chr.start();
    pi_student = reinterpret_cast<StudentWork2*>(student)->run(number_of_threads);
    chr.stop();
    if( verbose ) {
        std::cout << "\tStudent's Work Done in " << chr.elapsedTimeInMilliSeconds() << " ms" << std::endl;
        std::cout << "\tThe calculated value of PI is: " << pi_student<< std::endl;
    }
}


bool Exercise2::check() {
    return ( abs(pi_student - M_PI) < 1e-5 );
}


#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise1/Exercise1.h>

namespace {
}

// ==========================================================================================
void Exercise1::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << std::endl;
}

// ==========================================================================================
void Exercise1::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise1::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise1& Exercise1::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    return *this;
}

void Exercise1::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Test the student work" << std::endl;
    ChronoCPU chr;
    chr.start();
    reinterpret_cast<StudentWork1*>(student)->run();
    chr.stop();
    if( verbose )
        std::cout << "\tStudent's Work Done in " << chr.elapsedTimeInMilliSeconds() << " ms" << std::endl;
}


bool Exercise1::check() {
    return true;
}


#pragma warning( disable : 4244 ) 

#include <iostream>
#include <Exercise.h>
#include <exercise1/ExerciseImpl.h>

namespace {
}

// ==========================================================================================
void ExerciseImpl::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << std::endl;
}

// ==========================================================================================
void ExerciseImpl::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void ExerciseImpl::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
ExerciseImpl& ExerciseImpl::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    return *this;
}

void ExerciseImpl::run(const bool verbose) {    
    if( verbose ) {
        std::cout << std::endl << "Test the student work" << std::endl;
    }
    execute_and_display_time(false, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run();
    }, 1);
}


bool ExerciseImpl::check() {
    return true;
}


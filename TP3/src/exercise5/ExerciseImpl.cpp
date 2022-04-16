#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <exercise5/ExerciseImpl.h>
#include <immintrin.h>
#include <cstdlib>
#include <OPP.h>
#include <thread>

unsigned OPP::nbThreads = std::thread::hardware_concurrency();

// ==========================================================================================
void ExerciseImpl::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " [ -s=size ]" << std::endl 
        << "where -s  specifies the size of array (default is "<<size_of_input<<")."
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
    if( checkCmdLineFlag(argc, argv, "s") ) {
        unsigned value = getCmdLineArgumentInt(argc, argv, "s");
        if( value >  0 )
            size_of_input = value;
        else
            usageAndExit(argv[0], -1);  
    }
    return *this;
}

void ExerciseImpl::prepare_data() 
{    
    v_input.resize(size_of_input);
    long long i = 0;
    for(auto& v : v_input) 
        v = i++;
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const uint32_t nbTrys = 5;
    if( verbose ) {
        std::cout << "Student code will run "<<nbTrys<<" times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<(size_of_input)<<" integers ..." << std::endl;
    }
    execute_and_display_time(verbose, [&]() {
        v_output = reinterpret_cast<StudentWorkImpl*>(student)->run_reduce(v_input);
    }, nbTrys);
}


bool ExerciseImpl::check() 
{ 
    long long truth = size_of_input * (size_of_input-1) / 2;
    std::cout << "truth is " << truth << ", while student value is " << v_output << std::endl;
    if( v_output != truth ) 
        return false;
    return true;
}


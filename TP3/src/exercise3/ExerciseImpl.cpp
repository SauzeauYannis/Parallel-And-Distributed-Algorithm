#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <exercise3/ExerciseImpl.h>
#include <immintrin.h>
#include <cstdlib>
#include <thread>
#include <OPP.h>

unsigned OPP::nbThreads = std::thread::hardware_concurrency();

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
        << " [ -s=size ]" << std::endl 
        << "where -s  specifies the size of the arrays/vectors (default is "<<size_of_arrays<<")."
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
            size_of_arrays = value;
        else
            usageAndExit(argv[0], -1);  
    }
    if( checkCmdLineFlag(argc, argv, "s") ) {
        unsigned value = getCmdLineArgumentInt(argc, argv, "t");
        if( value > 0 && value < 2049 )
            OPP::nbThreads = value;
    }
    return *this;
}

void ExerciseImpl::prepare_data() 
{
    v_input_a.resize(size_of_arrays);
    v_input_b.resize(size_of_arrays);
    v_output_square.resize(size_of_arrays);
    v_output_sum.resize(size_of_arrays);
    for(auto i=size_of_arrays; i--;) {
        v_input_a[i] = int(i);
        v_input_b[i] = -int(i);
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const uint32_t nbTrys = 5;
    if( verbose ) {
        std::cout << "Number of threads: "<<OPP::nbThreads<<std::endl;
        std::cout << "Student code will run " << nbTrys << " times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" integers ..." << std::endl;
        std::cout << std::endl << "- square ..." << std::endl;        
    }
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run_square(v_input_a, v_output_square);
    }, nbTrys);
    if( verbose )
        std::cout << std::endl << "- sum of two arrays ..." << std::endl;        
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run_sum(v_input_a, v_input_b, v_output_sum);
    }, nbTrys);
}


bool ExerciseImpl::check() {
    for(auto i=size_of_arrays; i--; )
        if(v_input_a[i]*v_input_a[i] != v_output_square[i])
            return false;
    for(auto i=size_of_arrays; i--; ) {
        if(v_output_sum[i] != 0)
            return false;
    }
    return true;
}


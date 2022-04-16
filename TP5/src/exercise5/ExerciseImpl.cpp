#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include "ExerciseImpl.h"
#include <thread>
#include <OPP.h>

unsigned OPP::nbThreads = std::thread::hardware_concurrency();

namespace{

    template<typename T>
    void display_vector(std::vector<T>& vector, char const*const msg) {
        std::cout << msg;
        for(auto i :vector)
            std::cout << i << " ";
        std::cout << std::endl;
    }

    template<typename T>
    void display_vector_on_error(std::vector<T>& vector) 
    {
        if( vector.size() <= 17) {
            display_vector(vector, "--> bad job, you obtain: ");        
        }
    }
};

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
    return *this;
}

void ExerciseImpl::prepare_data() 
{
    v_input.resize(size_of_arrays);
    v_output.resize(size_of_arrays);
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_int_distribution<unsigned> distribution(0,unsigned(size_of_arrays));
    for(auto i=size_of_arrays; i--;) {
        v_input[i] = distribution(generator);
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const unsigned nbTry = 4u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" unsigned integers ..." << std::endl;
        std::cout << "- parallel radix sort ..." << std::endl; 
    }
    execute_and_display_time(verbose,[&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_radixSort_parallel(v_input, v_output);
    }, nbTry);
}


bool ExerciseImpl::check() {
    std::cout << "Verify the radix sort" << std::endl;
    if( size_of_arrays < 17 ) {
        display_vector(v_input,  " Input = ");
        display_vector(v_output, "Output = ");
    }
    std::vector<unsigned> truth(v_input);
    std::sort(truth.begin(), truth.end());
    for(size_t i=0; i<size_of_arrays; ++i) 
    {
        if( v_output[i] != truth[i] ) {
            std::cout << "-> result is not sorted ..." << std::endl;
            return false;
        }
    }
    std::cout << "-> radix sort is OK" << std::endl;
    return true;
}


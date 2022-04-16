#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <chronoCPU.hpp>
#include "ExerciseImpl.h"
#include <random>

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

int ExerciseImpl::random() 
{
    return randomDistribution(randomGenerator);
}

void ExerciseImpl::prepare_data() 
{
    v_a.resize(size_of_arrays);
    v_b.resize(size_of_arrays);
    v_output.resize(size_of_arrays);
    v_truth.resize(size_of_arrays);
    for(auto i=size_of_arrays; i--;) {
        v_a[i] = random();
        v_b[i] = random();
    }
}

void ExerciseImpl::prepare_truth()
{
    ChronoCPU chr;
    chr.start();
    for(auto i=size_of_arrays; i--;) {
            v_truth[i] = v_a[i] + v_b[i];
    }
    chr.stop();
    std::cout << " - Computation on CPU done in ";
    display_time( chr.elapsedTimeInMicroSeconds());
    std::cout << std::endl;
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const unsigned nbTry = 10u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" integers ..." << std::endl;
    }

    const unsigned size = unsigned(size_of_arrays);
	OPP::CUDA::DeviceBuffer<int> dev_a(v_a.data(), size);
	OPP::CUDA::DeviceBuffer<int> dev_b(v_b.data(), size);
	OPP::CUDA::DeviceBuffer<int> dev_result(size);

    execute_and_display_GPU_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_binary_map(dev_a, dev_b, dev_result);
    }, nbTry);

	dev_result.copyToHost(v_output.data());
}


bool ExerciseImpl::check() 
{
    std::cout << "Verify binary MAP calculations" << std::endl;
    prepare_truth();
    bool ok = true;
    for(auto i=size_of_arrays; i--; )
        if( v_truth[i] != v_output[i] ) {
            ok = false;
            break;
        }
    if ( !ok ) {
        if( size_of_arrays < 33 ) {
            std::cout << "-> sequential is WRONG ..." << std::endl;
            display_vector(v_a, "A = ");
            display_vector(v_b, "B = ");
            display_vector(v_output, "Result = ");
        }
        return false;
    }
    std::cout << "-> binary MAP is OK" << std::endl;
    return true;
}


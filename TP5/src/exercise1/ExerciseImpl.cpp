#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include "ExerciseImpl.h"

#include "OPP.h"

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
    v_predicate.resize(size_of_arrays);
    v_output_seq.resize(size_of_arrays);
    for(auto i=size_of_arrays; i--;) {
        v_input[i] = int(i);
        v_predicate[i] = bool( (i>>2) & 1 );
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const unsigned nbTry = 10u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" integers ..." << std::endl;
        std::cout << "- sequential ..." << std::endl;        
    }
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_partition_sequential(v_input, v_predicate, v_output_seq);
    }, nbTry);
}


bool ExerciseImpl::check() {
    std::cout << "Verify sequential calculations" << std::endl;
    if ( !check_sequential_True() || !check_sequential_False() ) {
        if( size_of_arrays < 17 ) {
            std::cout << "-> sequential is WRONG ..." << std::endl;
            display_vector(v_input, "I = ");
            display_vector(v_predicate, "P = ");
            display_vector(v_output_seq, "O = ");
        }
        return false;
    }
    std::cout << "-> sequential is OK" << std::endl;
    return true;
}


bool ExerciseImpl::check_sequential_True() 
{
    size_t outputIndex = 0;
    for(auto i=0; i<size_of_arrays; ++i) {
        if( v_predicate[i] == 0 ) continue;
        if( v_output_seq[outputIndex] != v_input[i])
            return false;
        ++outputIndex;
    }
    return true;
}


bool ExerciseImpl::check_sequential_False() 
{
    size_t outputIndex = size_of_arrays;
    for(auto i=size_of_arrays; i--;) {
        if( v_predicate[i] != 0 ) continue;
        --outputIndex;
        if( v_output_seq[outputIndex] != v_input[i])
            return false;
    }
    return true;
}

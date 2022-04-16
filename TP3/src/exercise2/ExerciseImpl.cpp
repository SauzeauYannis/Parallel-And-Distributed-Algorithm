#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <exercise2/ExerciseImpl.h>
#include <immintrin.h>
#include <cstdlib>

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
    return *this;
}

void ExerciseImpl::prepare_data() 
{
    v_input.resize(size_of_arrays);
    for(auto i=size_of_arrays; i--;) {
        v_input[i] = i;;
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const uint32_t nbTrys = 5;
    if( verbose ) {
        std::cout << "Student code will run "<<nbTrys<<" times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" integers ..." << std::endl;
        std::cout << std::endl << "- sum ..." << std::endl;        
    }
    execute_and_display_time(verbose, [&]() {
        output_sum = reinterpret_cast<StudentWorkImpl*>(student)->run_sum(v_input);
    }, nbTrys);
    if( verbose )
        std::cout << std::endl << "- sum of square ..." << std::endl;        
    execute_and_display_time(verbose, [&]() {
        output_sum_square = reinterpret_cast<StudentWorkImpl*>(student)->run_sum_square(v_input);
    });
    if( verbose )
        std::cout << std::endl << "- optimized sum of square ..." << std::endl;        
    execute_and_display_time(verbose, [&]() {
        output_sum_square_opt = reinterpret_cast<StudentWorkImpl*>(student)->run_sum_square_opt(v_input);
    }, nbTrys);
}


bool ExerciseImpl::check() 
{
    const long long sum = size_of_arrays * (size_of_arrays-1) / 2;
    std::cout << "wait "<<sum<<", and receive "<<output_sum<<std::endl;
    if (output_sum != sum)
        return false;
    const long long sum_square = (size_of_arrays-1) * size_of_arrays * (2*size_of_arrays-1) / 6;
    std::cout << "wait "<<sum_square<<", and receive "<<output_sum_square<<std::endl;
    if(output_sum_square != sum_square)
        return false;
    std::cout << "wait "<<sum_square<<", and receive "<<output_sum_square_opt<<" after optimization"<<std::endl;
    if(output_sum_square != sum_square)
        return false;
    return true;
}


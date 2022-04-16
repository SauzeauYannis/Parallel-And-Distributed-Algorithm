#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <exercise4/ExerciseImpl.h>
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
    v_input.resize(size_of_arrays);
    v_map.resize(size_of_arrays);
    v_output_gather.resize(size_of_arrays);
    v_output_scatter.resize(size_of_arrays);
    auto middle = (size_of_arrays+1)/2;
    for(auto i=size_of_arrays; i--;) {
        v_input[i] = int(i); // 0, 1, ....
        v_map[i] = size_t(i<middle ? 2*i : (i-middle)*2+1); // 0, 2, 4 ... 2*(middle-2) ; 1, 3, .... //
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const uint32_t nbTrys = 5;
    if( verbose ) {
        std::cout << "Number of threads: "<<OPP::nbThreads<<std::endl;
        std::cout << "Student code will run "<<nbTrys<<" times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" integers ..." << std::endl;
        std::cout << std::endl << "- gather ..." << std::endl;        
    }
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run_gather(v_input, v_map, v_output_gather);
    }, nbTrys);
    if( verbose )
        std::cout << std::endl << "- scatter ..." << std::endl;        
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run_scatter(v_input, v_map, v_output_scatter);
    }, nbTrys);
}


bool ExerciseImpl::check() {
    if( size_of_arrays <= 16 ) {
        std::cout << "v_input = {";
        for(auto& v : v_input)  {
            std::cout << v << ",";            
        }
        std::cout << std::endl;
        std::cout << "v_map = {";
        for(auto& v : v_map)  {
            std::cout << v << ",";   
        }
        std::cout << std::endl;
        std::cout << "v_gather = {";
        for(auto& v : v_output_gather)  {
            std::cout << v << ",";        
        }
        std::cout << std::endl;
        std::cout << "v_scatter = {";
        for(auto& v : v_output_scatter)  {
            std::cout << v << ",";           
        }
        std::cout << std::endl;
    }
    // check gather
    for(auto i=v_input.size(); i--;)
        if( v_output_gather[i] != v_input[v_map[i]])
            return false;
    // check scatter
    for(auto i=v_input.size(); i--;)
        if( v_output_scatter[v_map[i]] != v_input[i])
            return false;
    return true;
}


#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include "ExerciseImpl.h"

#include "OPP.h"

unsigned OPP::nbThreads = std::thread::hardware_concurrency();

namespace{

    template<typename T>
    void display_vector_on_error(std::vector<T>& vector) 
    {
        if( vector.size() <= 17) {
            std::cout << "--> bad job, you obtain: ";
            for(auto i :vector)
                std::cout << i << " ";
            std::cout << std::endl;
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
        << " [ -s=size ] [ -s=exponent ]" << std::endl 
        << "where -s  specifies the size of the arrays/vectors (default is "<<size_of_arrays<<")." << std::endl
        << "where -e  specifies the logarithm of the arrays/vector' size (default is "<<uint32_t(std::log2(size_of_arrays))<<")." << std::endl
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
    if( checkCmdLineFlag(argc, argv, "e") ) {
        unsigned value = getCmdLineArgumentInt(argc, argv, "e");
        if( value >  0 && value < 64 )
            size_of_arrays = 1LLu << value;
        else
            usageAndExit(argv[0], -1);  
    }
    return *this;
}

void ExerciseImpl::prepare_data() 
{
    v_input.resize(size_of_arrays);
    v_output_seq.resize(size_of_arrays);
    v_output_par.resize(size_of_arrays);
    for(auto i=size_of_arrays; i--;) {
        v_input[i] = 1;
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const unsigned nbTry = 4u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<size_of_arrays<<" integers ..." << std::endl;
        std::cout << "- sequential ..." << std::endl;        
    }
    std::function<int(int,int)> plus = [](int a,int b)->int{return a+b;};
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_scan_sequential(v_input, v_output_seq, int(0), plus);
    });
    if( verbose )
        std::cout << "- parallel with " << OPP::nbThreads << " threads ..." << std::endl;        
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)
            ->run_scan_parallel(v_input, v_output_par, int(0), plus);
    });
}


bool ExerciseImpl::check() 
{
    bool seq_ok = check_sequential();
    bool par_ok = check_parallel();
    bool ass_ok = check_associativity();
    return seq_ok && par_ok && ass_ok;
}


bool ExerciseImpl::check_sequential()
{
    std::cout << "Verify SEQUENTIAL calculations" << std::endl;
    int sum = 0;
    for(auto i=0; i<size_of_arrays; ++i) {
        if(sum != v_output_seq[i]) {
            display_vector_on_error(v_output_seq);
            return false;
        }
        sum += v_input[i];
    }
    std::cout << "-> sequential is OK" << std::endl;    
    return true;
}

bool ExerciseImpl::check_parallel() 
{
    std::cout << "Verify PARALLEL calculations" << std::endl;
    for(auto i=size_of_arrays; i--; ) {
        if(v_output_seq[i] != v_output_par[i]) {
            display_vector_on_error(v_output_par);
            return false;
        }
    }
    std::cout << "-> parallel is OK" << std::endl;
    return true;
}


bool ExerciseImpl::check_associativity()
{
    // is the pattern associative? Check sequential, then parallel
    std::cout << "Verify ASSOCIATIVITY" << std::endl;
    const size_t vSize = 8;
    std::vector<iMatrix22> input = init_check_associativity(vSize);
    if( !check_associativity_seq(input) ) 
        return false;
    if( !check_associativity_par(input) ) 
        return false;
    std::cout << "-> associativity is OK" << std::endl;
    return true;
}


std::vector<iMatrix22> ExerciseImpl::init_check_associativity(size_t size)
{
    std::vector<iMatrix22> input(size);
    for(int i=0; i<size; ++i) {
        input[i].fill(i+1);
    }
    return input;
}

bool ExerciseImpl::check_associativity_seq(std::vector<iMatrix22>&input)
{
    std::vector<iMatrix22> output(input.size());
    std::function<iMatrix22(iMatrix22,iMatrix22)> multiply = [](iMatrix22 a,iMatrix22 b)->iMatrix22{
        return a * b;
    } ;
    reinterpret_cast<StudentWorkImpl*>(student)->run_scan_sequential(input, output, iMatrix22::make_identity(), multiply);
    iMatrix22 expected = iMatrix22::make_identity();
    for(int i=0; i<input.size(); ++i) {
        if( expected != output[i] ) {
            std::cout << "--> failed at output[" << i << "]:" << std::endl;
            std::cout << "---> expected " << expected << std::endl;
            std::cout << "---> get      " << output[i] << std::endl;
            return false;
        }
        expected = expected * input[i];
    }
    std::cout << "--> associativity OK for SEQUENTIAL inclusive SCAN" << std::endl;    
    return true;
}


bool ExerciseImpl::check_associativity_par(std::vector<iMatrix22>&input)
{
    std::vector<iMatrix22> output(input.size());
    std::function<iMatrix22(iMatrix22,iMatrix22)> multiply = [](iMatrix22 a,iMatrix22 b)->iMatrix22{
        return a * b;
    } ;
    reinterpret_cast<StudentWorkImpl*>(student)->run_scan_parallel(input, output, iMatrix22::make_identity(), multiply);
    iMatrix22 expected = iMatrix22::make_identity();
    for(int i=0; i<input.size(); ++i) {
        if( expected != output[i] ) {
            std::cout << "--> failed at output[" << i << "]:" << std::endl;
            std::cout << "---> expected " << expected << std::endl;
            std::cout << "---> get      " << output[i] << std::endl;
            return false;            
        }
        expected = expected * input[i];
    }
    std::cout << "--> associativity OK for PARALLEL inclusive SCAN" << std::endl;
    return true;
}

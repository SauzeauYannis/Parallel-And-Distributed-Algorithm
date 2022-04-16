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
        << "where -s  specifies the number of AVX registers (default is "<<number_of_avx_registers<<")."
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
            number_of_avx_registers = value;
        else
            usageAndExit(argv[0], -1);  
    }
    return *this;
}

void ExerciseImpl::prepare_data() {
    #ifdef WIN32
    input_floats = static_cast<float*>(_aligned_malloc((number_of_avx_registers << 3) * sizeof(float),32));
    student_floats = static_cast<float*>(_aligned_malloc( (number_of_avx_registers << 3) * sizeof(float),32));
    input_m256s = static_cast<__m256*>(_aligned_malloc(number_of_avx_registers * sizeof(__m256),32));
    student_m256s = static_cast<__m256*>(_aligned_malloc(number_of_avx_registers * sizeof(__m256),32));
    #else
    input_floats = static_cast<float*>(std::aligned_alloc(32, (number_of_avx_registers << 3) * sizeof(float)));
    student_floats = static_cast<float*>(std::aligned_alloc(32, (number_of_avx_registers << 3) * sizeof(float)));
    input_m256s = static_cast<__m256*>(std::aligned_alloc(32, number_of_avx_registers * sizeof(__m256)));
    student_m256s = static_cast<__m256*>(std::aligned_alloc(32, number_of_avx_registers * sizeof(__m256)));
    #endif
    for(int i=(number_of_avx_registers<<3); i--;) {
        input_floats[i] = float(i);
    }
    for(int i=number_of_avx_registers; i--;){
        input_m256s[i] = _mm256_load_ps(&input_floats[i<<3]);
    }
}

void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    if( verbose ) {
        std::cout << "Student code will run 20 times for statistics ..." << std::endl;
        std::cout << std::endl << "Test the student work with "<<(number_of_avx_registers<<3)<<" floats ..." << std::endl;
    }
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run(input_floats, student_floats, number_of_avx_registers << 3);
    });
    if( verbose )
        std::cout << std::endl << "Test the student work with "<<(number_of_avx_registers)<<" AVX floats ..." << std::endl;
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run(input_m256s, student_m256s, number_of_avx_registers);
    });
}


bool ExerciseImpl::check() {
    for(auto i=(number_of_avx_registers*8); i--; )
        if(sqrtf(input_floats[i]) != student_floats[i])
            return false;
    for(auto i=number_of_avx_registers; i--; ) {
        const __m256 cmp = 
            _mm256_cmp_ps(_mm256_sqrt_ps(input_m256s[i]), student_m256s[i], _CMP_NEQ_OQ);
        if(_mm256_movemask_ps(cmp) != 0)
            return false;
    }
    return true;
}


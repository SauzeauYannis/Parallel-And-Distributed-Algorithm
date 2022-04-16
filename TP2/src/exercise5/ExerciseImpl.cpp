#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <exercise5/ExerciseImpl.h>
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
        if( value >  0 && value <= 8'192)
            number_of_avx_registers = value;
        else
            usageAndExit(argv[0], -1);  
    }
    return *this;
}

void ExerciseImpl::prepare_data() 
{    
    auto row_size = (number_of_avx_registers << 3);
    auto matrix_size = number_of_avx_registers * row_size;
    auto matrix_size64 = matrix_size << 3;
    #ifdef WIN32
    vector_floats = static_cast<float*>(_aligned_malloc(row_size * sizeof(float),32));
    vector_m256s = static_cast<__m256*>(_aligned_malloc(number_of_avx_registers * sizeof(__m256),32));
    input_floats = static_cast<float*>(_aligned_malloc(matrix_size64 * sizeof(float),32));
    student_floats = static_cast<float*>(_aligned_malloc(row_size * sizeof(float),32));
    input_m256s = static_cast<__m256*>(_aligned_malloc(matrix_size * sizeof(__m256),32));
    student_m256s = static_cast<float*>(_aligned_malloc(number_of_avx_registers * sizeof(__m256),32));
    #else
    vector_floats = static_cast<float*>(std::aligned_alloc(32, row_size * sizeof(float)));
    vector_m256s = static_cast<__m256*>(std::aligned_alloc(32, number_of_avx_registers * sizeof(__m256)));
    input_floats = static_cast<float*>(std::aligned_alloc(32, matrix_size64 * sizeof(float)));
    student_floats = static_cast<float*>(std::aligned_alloc(32, row_size * sizeof(float)));
    input_m256s = static_cast<__m256*>(std::aligned_alloc(32, matrix_size * sizeof(__m256)));
    student_m256s = static_cast<float*>(std::aligned_alloc(32, row_size * sizeof(float)));
    #endif
    for(auto i=row_size; i--;) {
        vector_floats[i] = float(i);
        for(auto j= row_size * i; j< row_size*(i+1); ++j)
            input_floats[j] = float(i);
    }
    for(auto i=number_of_avx_registers; i--;){
        vector_m256s[i] = _mm256_load_ps(&vector_floats[i<<3]);
    }
    for(auto i=number_of_avx_registers*row_size; i--;){
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
        reinterpret_cast<StudentWorkImpl*>(student)->run(input_floats, vector_floats, student_floats, number_of_avx_registers << 3);
    });
    if( verbose )
        std::cout << std::endl << "Test the student work with "<<(number_of_avx_registers)<<" AVX floats ..." << std::endl;
    execute_and_display_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->run(input_m256s, vector_m256s, student_m256s, number_of_avx_registers);
    });
}


bool ExerciseImpl::check() 
{ 
    auto vector_size = number_of_avx_registers << 3;
    for(auto i=vector_size; i--; ) 
    {
        const float truth = float(i * vector_size * (vector_size-1) / 2);
        const float mm_value = student_m256s[i];
        if( student_floats[i] != truth || mm_value != truth ) 
            return false;
    }
    return true;
}

#pragma once

#include <Exercise.h>
#include <immintrin.h>
#include <vector>
#include <exo5/student.h>

class ExerciseImpl : public Exercise 
{
public:
    ExerciseImpl(const std::string& name ) 
        : Exercise(name, new StudentWorkImpl()), number_of_avx_registers(1<<4)
    {}

    ExerciseImpl() = delete;
    ExerciseImpl(const ExerciseImpl&) = default;
    ~ExerciseImpl() = default;
    ExerciseImpl& operator= (const ExerciseImpl&) = delete;

    ExerciseImpl& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    

    unsigned number_of_avx_registers;

    float* vector_floats;
    __m256* vector_m256s = nullptr;

    float* student_floats;
    float* student_m256s;
    
    float* input_floats;
    __m256* input_m256s = nullptr;

    void prepare_data();

};
#pragma once

#include <Exercise.h>
#include <immintrin.h>
#include <vector>
#include <exo2/student.h>

class ExerciseImpl : public Exercise 
{
public:
    ExerciseImpl(const std::string& name ) 
        : Exercise(name, new StudentWorkImpl()), number_of_avx_registers(1<<10)
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

    float* student_floats = nullptr;
    __m256* student_m256s;
    float* input_floats = nullptr;
    __m256* input_m256s;

    void prepare_data();

};
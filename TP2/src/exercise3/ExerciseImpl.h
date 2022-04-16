#pragma once

#include <Exercise.h>
#include <immintrin.h>
#include <vector>
#include <exo3/student.h>

class ExerciseImpl : public Exercise 
{
public:
    ExerciseImpl(const std::string& name ) 
        : Exercise(name, new StudentWorkImpl()), number_of_avx_registers(1<<10), number_of_threads(1)
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

    float* student_floats;
    __m256* student_m256s = nullptr;
    float* input_floats;
    __m256* input_m256s = nullptr;

    size_t number_of_threads;

    void prepare_data();

};
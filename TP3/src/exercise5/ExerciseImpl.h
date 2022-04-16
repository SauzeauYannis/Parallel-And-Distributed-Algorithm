#pragma once

#include <Exercise.h>
#include <immintrin.h>
#include <vector>
#include <exo5/student.h>

class ExerciseImpl : public Exercise 
{
public:
    ExerciseImpl(const std::string& name ) 
        : Exercise(name, new StudentWorkImpl()), size_of_input(1024)
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

    size_t size_of_input;

    std::vector<long long> v_input;
    long long v_output;
    
    void prepare_data();

};
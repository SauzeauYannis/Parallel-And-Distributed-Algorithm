#pragma once

#include <Exercise.h>
#include <exo2/student.h>
#include <vector>

class ExerciseImpl : public Exercise 
{
public:
    ExerciseImpl(const std::string& name ) 
        : Exercise(name, new StudentWorkImpl()), size_of_arrays(1024)
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

    void prepare_data();

    size_t size_of_arrays;

    std::vector<long long> v_input;
    
    long long output_sum_square_opt;
    long long output_sum_square;
    long long output_sum;
};
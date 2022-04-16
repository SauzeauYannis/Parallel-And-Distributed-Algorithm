#pragma once

#include <Exercise.h>
#include <exo3/student.h>
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

    std::vector<int> v_input_a;
    std::vector<int> v_input_b;
    
    std::vector<int> v_output_square;
    std::vector<int> v_output_sum;
};
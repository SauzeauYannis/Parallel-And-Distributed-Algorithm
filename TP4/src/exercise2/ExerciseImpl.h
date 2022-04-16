#pragma once

#include <Exercise.h>
#include <exo2/student.h>
#include <vector>
#include <iMatrix22.h>

class ExerciseImpl : public Exercise 
{
public:
    ExerciseImpl(const std::string& name ) 
        : Exercise(name, new StudentWorkImpl()), size_of_arrays(1024)
    {}

    ExerciseImpl() = delete;
    ExerciseImpl(const ExerciseImpl&) = default;
    ~ExerciseImpl() = default;
    ExerciseImpl& operator= (const ExerciseImpl&) = default;

    ExerciseImpl& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    

    void prepare_data();

    size_t size_of_arrays;

    std::vector<int> v_input;
    
    std::vector<int> v_output_seq;
    std::vector<int> v_output_par;

    bool check_sequential();
    bool check_parallel();
    bool check_associativity();
    std::vector<iMatrix22> init_check_associativity(const size_t);
    bool check_associativity_seq(std::vector<iMatrix22>&);
    bool check_associativity_par(std::vector<iMatrix22>&);
};
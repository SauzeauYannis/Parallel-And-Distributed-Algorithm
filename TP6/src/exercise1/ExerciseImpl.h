#pragma once

#include <ExerciseGPU.h>
#include <exo1/student.h>
#include <vector>
#include <random>

class ExerciseImpl : public ExerciseGPU
{
public:
    ExerciseImpl(const std::string& name ) 
        : ExerciseGPU(name, new StudentWorkImpl()), size_of_arrays(1024)
    {
        std::random_device rd;
        randomGenerator = std::mt19937(rd());
        randomDistribution = std::uniform_int_distribution<int>(0, 1<<16);
    }

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
    void prepare_truth();
    
    size_t size_of_arrays;

    std::vector<int> v_a;
    std::vector<int> v_b;

    std::vector<int> v_output;
    std::vector<int> v_truth;

    int random();
    std::uniform_int_distribution<int> randomDistribution;
    std::mt19937 randomGenerator;
};
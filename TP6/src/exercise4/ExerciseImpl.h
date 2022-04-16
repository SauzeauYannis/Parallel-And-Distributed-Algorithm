#pragma once

#include <ExerciseGPU.h>
#include <exo4/student.h>
#include <ppm.h>
#include <vector>
#include <random>


class ExerciseImpl : public ExerciseGPU
{
public:
    ExerciseImpl(const std::string& name ) 
        : ExerciseGPU(name, new StudentWorkImpl())
    {
        std::random_device rd;
        randomGenerator = std::mt19937(rd());
        randomDistribution = std::uniform_int_distribution<int>(0, 9-1);
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
    void prepare_image();
    void prepare_map();

    PPMBitmap *sourceImage;
    PPMBitmap *destImage;

    std::vector<uchar2> v_map;

    char* inputFileName;
    
    int random();
    std::uniform_int_distribution<int> randomDistribution;
    std::mt19937 randomGenerator;
};
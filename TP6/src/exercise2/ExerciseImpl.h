#pragma once

#include <ExerciseGPU.h>
#include <exo2/student.h>
#include <ppm.h>

class ExerciseImpl : public ExerciseGPU
{
public:
    ExerciseImpl(const std::string& name ) 
        : ExerciseGPU(name, new StudentWorkImpl()), borderWidth(1)
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

    PPMBitmap *sourceImage;
    PPMBitmap *destImage;

    unsigned borderWidth;

    char* inputFileName;
};
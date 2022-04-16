#pragma once

#include <ExerciseGPU.h>
#include <exo5/student.h>
#include <ppm.h>
#include <vector>


class ExerciseImpl : public ExerciseGPU
{
public:
    ExerciseImpl(const std::string& name ) 
        : ExerciseGPU(name, new StudentWorkImpl()), filterWidth(7)
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
    void prepare_image();
    float gaussian(float sigma, float x, float y);
    void prepare_filter();
    void set_gaussian_filter();
    void normalize_filter();

    PPMBitmap *sourceImage;
    PPMBitmap *destImage;

    std::vector<float> v_filter;
    int filterWidth;

    char* inputFileName;
};
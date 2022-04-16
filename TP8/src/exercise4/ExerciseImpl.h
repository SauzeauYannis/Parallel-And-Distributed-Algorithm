#pragma once

#include <ExerciseGPU.h>
#include <exo4/student.h>
#include <vector>
#include <random>
#include <ppm.h>
#include <reference/ImageEqualizer.h>

class ExerciseImpl : public ExerciseGPU
{
public:
    ExerciseImpl(const std::string& name ) 
        : ExerciseGPU(name, new StudentWorkImpl()), 
        sourceImage(nullptr), destImage(nullptr),
        inputFileName(nullptr), reference(nullptr)
    {
    }

    ExerciseImpl() = delete;
    ExerciseImpl(const ExerciseImpl&) = default;
    ~ExerciseImpl() = default;
    ExerciseImpl& operator= (const ExerciseImpl&) = default;

    ExerciseImpl& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    bool check_Transformation();
    bool checkImagesAreEquals(const PPMBitmap&, const PPMBitmap&);
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    

    void prepare_data();
    void prepare_truth();

    std::string makeFileName(const char*fileName, const std::string& extension);
    void saveTo(const std::string& fileName, PPMBitmap& image);

    PPMBitmap *sourceImage;
    PPMBitmap *destImage;
    
    char* inputFileName;

    std::vector<float> Transformation;

    ImageEqualizer* reference;
};
#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <chronoCPU.hpp>
#include "ExerciseImpl.h"
#include <random>
#include <ppm.h>
#include <utility>
#include <cmath>

#ifndef M_PI
constexpr float M_PI = 3.14159265358979323846f;
#endif

namespace{

};

// ==========================================================================================
void ExerciseImpl::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " [ -i=<input.ppm> ] [ -f=<filter half size> ]" << std::endl 
        << "where -i  specifies the file name of the input image," << std::endl
        << "  and -f  specifies the half-size of the filter."
        << std::endl;
}

// ==========================================================================================
void ExerciseImpl::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void ExerciseImpl::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
ExerciseImpl& ExerciseImpl::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    std::cout << "argv[1] = " << argv[1] << std::endl;
    if( checkCmdLineFlag(argc, argv, "i") ) {
        if( !getCmdLineArgumentString(argc, argv, "i", &inputFileName) ) 
        {
            std::cerr << "unable to get the input file name" << std::endl;
            usageAndExit(argv[0], -1);  
        }        
    }
    if( checkCmdLineFlag(argc, argv, "f") ) {
        const int arg = getCmdLineArgumentInt(argc, argv, "f");
        if( arg < 0 ) 
        {
            std::cerr << "Filter half size must be greater or equal to zero" << std::endl;
            usageAndExit(argv[0], -1);
        }
        filterWidth = 1 + 2*arg;
    }
    return *this;
}


void ExerciseImpl::prepare_image() 
{
    std::cout << "- load input file " << inputFileName << std::endl;
    sourceImage = new PPMBitmap(inputFileName);
    const unsigned width = sourceImage->getWidth();
    const unsigned height = sourceImage->getHeight();
    std::cout << "- File <" << inputFileName << "> loaded. Contains "<<width<< " per " << height << " pixels." << std::endl;
    destImage = new PPMBitmap(width, height);
}


float ExerciseImpl::gaussian(float sigma, float x, float y)
{
    const float twoSquareSigma = 2.f * sigma * sigma;
    const float squareDistance = x*x + y*y;
    const float exponent = expf( - squareDistance / twoSquareSigma);
    const float factor = 1.f / M_PI / twoSquareSigma;
    return factor * exponent; 
}


void ExerciseImpl::set_gaussian_filter()
{
    const float sigma = float(filterWidth/2);
    for(int i=0; i<filterWidth; ++i)
        for(int j=0; j<filterWidth; ++j)
        {
            const float x = float(i-filterWidth/2);
            const float y = float(j-filterWidth/2);
            v_filter.push_back(gaussian(sigma, x, y));
        }
}

void ExerciseImpl::normalize_filter()
{
    float sum = 0.f;
    for(int i=0; i<filterWidth*filterWidth; ++i)
        sum += v_filter[i];
    for(int i=0; i<filterWidth*filterWidth; ++i)
        v_filter[i] /= sum;
}

void ExerciseImpl::prepare_filter()
{
    set_gaussian_filter();
    normalize_filter();
}

void ExerciseImpl::prepare_data() 
{
    prepare_image();
    prepare_filter();
}


void ExerciseImpl::run(const bool verbose) 
{    
    prepare_data();
    const unsigned nbTry = 10u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
    }

    const unsigned width = sourceImage->getWidth();
    const unsigned height = sourceImage->getHeight();
    const unsigned size = width * height;
	OPP::CUDA::DeviceBuffer<uchar3> dev_input(reinterpret_cast<uchar3*>(sourceImage->getPtr()), size);
	OPP::CUDA::DeviceBuffer<uchar3> dev_output(reinterpret_cast<uchar3*>(destImage->getPtr()), size);
    OPP::CUDA::DeviceBuffer<float> dev_filter(v_filter.data(), unsigned(v_filter.size()));
    
    execute_and_display_GPU_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_filter(dev_input, dev_output, dev_filter, width, height, filterWidth);
    }, nbTry);

	dev_output.copyToHost(reinterpret_cast<uchar3*>(destImage->getPtr()));
}


bool ExerciseImpl::check() 
{
    std::string s_output(inputFileName);
    size_t delimiter_pos = s_output.find_last_of('.');
    s_output.resize(delimiter_pos);
    s_output.append("_filtered.ppm");
    std::cout << "Save result into " << s_output << std::endl;
    destImage->saveTo(s_output.c_str());
    return true;
}


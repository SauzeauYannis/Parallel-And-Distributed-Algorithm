#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <chronoCPU.hpp>
#include "ExerciseImpl.h"
#include <random>
#include <ppm.h>

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
        << " [ -i=<input.ppm> ] [ -b=<border width> ]" << std::endl 
        << "where -i  specifies the file name of the input image," << std::endl
        << "  and -b  specifies the width of the thumbnail border (default 1)."
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
    if (checkCmdLineFlag(argc, argv, "b") ) {
        int width = getCmdLineArgumentInt(argc, argv, "b");
        if( width>0 )
            borderWidth = unsigned(width);
    }
    return *this;
}

void ExerciseImpl::prepare_data() 
{
    std::cout << "- load input file " << inputFileName << std::endl;
    sourceImage = new PPMBitmap(inputFileName);
    const unsigned width = sourceImage->getWidth();
    const unsigned height = sourceImage->getHeight();
    std::cout << "- File <" << inputFileName << "> loaded. Contains "<<width<< " per " << height << " pixels." << std::endl;
    destImage = new PPMBitmap(width, height);
}


void ExerciseImpl::run(const bool verbose) {    
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
	
    execute_and_display_GPU_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_thumbnail(dev_input, dev_output, make_uchar3(255, 255, 255), borderWidth, width, height);
    }, nbTry);

	dev_output.copyToHost(reinterpret_cast<uchar3*>(destImage->getPtr()));
}


bool ExerciseImpl::check() 
{
    std::string s_output(inputFileName);
    size_t delimiter_pos = s_output.find_last_of('.');
    s_output.resize(delimiter_pos);
    s_output.append("_thumbnail.ppm");
    std::cout << "Save result into " << s_output << std::endl;
    destImage->saveTo(s_output.c_str());
    return true;
}


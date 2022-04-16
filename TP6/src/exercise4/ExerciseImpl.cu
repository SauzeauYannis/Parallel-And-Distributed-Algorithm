#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <chronoCPU.hpp>
#include "ExerciseImpl.h"
#include <random>
#include <ppm.h>
#include <utility>


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
        << " [ -i=<input.ppm> ]" << std::endl 
        << "where -i  specifies the file name of the input image."
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
    return *this;
}

int ExerciseImpl::random() {
    return randomDistribution( randomGenerator ) ;
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

void ExerciseImpl::prepare_map()
{
    for(unsigned i=0; i<3; ++i)
        for(unsigned j=0; j<3; ++j)
            v_map.push_back(make_uchar2(i,j));
    for(unsigned i=1<<16; i--;)
        std::swap(v_map[random()], v_map[random()]);
}

void ExerciseImpl::prepare_data() 
{
    prepare_image();
    prepare_map();
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
    OPP::CUDA::DeviceBuffer<uchar2> dev_map(v_map.data(), unsigned(v_map.size()));
    
    execute_and_display_GPU_time(verbose, [&]() {
        reinterpret_cast<StudentWorkImpl*>(student)->
            run_thumbnail_scatter(dev_input, dev_output, dev_map, width, height);
    }, nbTry);

	dev_output.copyToHost(reinterpret_cast<uchar3*>(destImage->getPtr()));
}


bool ExerciseImpl::check() 
{
    std::string s_output(inputFileName);
    size_t delimiter_pos = s_output.find_last_of('.');
    s_output.resize(delimiter_pos);
    s_output.append("_scatter.ppm");
    std::cout << "Save result into " << s_output << std::endl;
    destImage->saveTo(s_output.c_str());
    return true;
}


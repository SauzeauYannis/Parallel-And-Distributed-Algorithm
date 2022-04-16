#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <random>
#include <ppm.h>
#include <chronoCPU.hpp>
#include <exercise1/ImageBlockEffect.h>
#include <exercise3/ExerciseImpl.h>
#include <exercise3/ExerciseRunner.h>

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
        << " [ -i=<input.ppm> ]  [ -w=nb_warps] " << std::endl 
        << "where -i  specifies the file name of the input image," << std::endl
        << "  and -w  specifies the number of warps ("<<nbWarps<<" per default)."
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
    if (checkCmdLineFlag(argc, argv, "w") ) {
        const int value = getCmdLineArgumentInt(argc, argv, "w"); 
        if( value < 1 || value > 32 || !isPowerOfTwo(value) )
            {
                std::cerr << "The number of warps must be a power of two in [1..32]" << std::endl;
                usageAndExit(argv[0], -1);  
            }
        nbWarps = value;
    }
    return *this;
}

bool ExerciseImpl::isPowerOfTwo(int value)
{
    int nb = 0;
    while(value) 
    {
        nb += (value & 1);
        value >>= 1;
    }
    return nb == 1;
}
    
void ExerciseImpl::prepare_data() 
{
    std::cout << "- load input file " << inputFileName << std::endl;
    sourceImage = new PPMBitmap(inputFileName);
    const unsigned width = sourceImage->getWidth();
    const unsigned height = sourceImage->getHeight();
    std::cout << "- File <" << inputFileName << "> loaded. Contains "<<width<< " per " << height << " pixels." << std::endl;
    destImage = new PPMBitmap(width, height);
    prepare_truth();
}


void ExerciseImpl::prepare_truth() 
{
    trustedImage = new PPMBitmap(sourceImage->getWidth(), sourceImage->getHeight());
    std::cout << "Calculate the reference on CPU ..." << std::endl;    
    const unsigned nbTry = 10u;
    execute_and_display_time(
        true, 
        [&]() { ImageBlockEffect(*sourceImage, *trustedImage).applyBlockEffet(); }, 
        nbTry,
        std::string("\tDone in")
    );
}


void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const unsigned nbTry = 25u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
    }

    ExerciseRunner runner(sourceImage, nbWarps);
	
    execute_and_display_GPU_time(verbose, [&]() {
        runner.run(reinterpret_cast<StudentWorkImpl*>(student));
    }, nbTry);

	runner.copyTo(destImage);
}


bool ExerciseImpl::check() 
{
    saveTo(makeFileName(inputFileName, "_block_reference.ppm"), *trustedImage);
    saveTo(makeFileName(inputFileName, "_block_student.ppm"), *destImage);
    return checkImagesAreEquals(*trustedImage, *destImage);
}

std::string ExerciseImpl::makeFileName(const char*fileName, const std::string& extension)
{
    std::string s_output(fileName);
    size_t delimiter_pos = s_output.find_last_of('.');
    s_output.resize(delimiter_pos);
    s_output.append(extension);
    return s_output;
}
    
void ExerciseImpl::saveTo(const std::string& fileName, PPMBitmap& image)
{
    std::cout << "Save result into " << fileName << std::endl;
    image.saveTo(fileName.c_str());
}

bool ExerciseImpl::checkImagesAreEquals(const PPMBitmap&imageA, const PPMBitmap&imageB)
{
    if( imageA.getWidth() != imageB.getWidth() || imageA.getHeight() != imageB.getHeight() )
        return false;
    const uchar* ptrA = imageA.getPtr();
    const uchar* ptrB = imageB.getPtr();
    for(unsigned i=imageA.getWidth()*imageA.getWidth(); i--;)
        if( std::max(ptrA[i],ptrB[i]) - std::min(ptrA[i],ptrB[i]) > 1 ) return false;
    return true;
}
    
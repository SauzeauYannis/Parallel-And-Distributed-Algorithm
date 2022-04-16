#include <iostream>
#include <exercise3/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 3
    ExerciseImpl("Exercise 3 : GATHER ta vignette")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

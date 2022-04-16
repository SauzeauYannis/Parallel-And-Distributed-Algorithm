#include <iostream>
#include <exercise2/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 2
    ExerciseImpl("Exercise 2 : effet vignette")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

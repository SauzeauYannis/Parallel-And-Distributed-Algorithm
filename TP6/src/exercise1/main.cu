#include <iostream>
#include <exercise1/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 1
    ExerciseImpl("Exercise 1 : MAP binaire")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

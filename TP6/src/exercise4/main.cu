#include <iostream>
#include <exercise4/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 4
    ExerciseImpl("Exercise 4 : SCATTER ta vignette")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

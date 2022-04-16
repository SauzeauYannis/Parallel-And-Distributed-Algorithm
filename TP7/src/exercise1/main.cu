#include <iostream>
#include <exercise1/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 1
    ExerciseImpl("Exercise 1 : simple block effet")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

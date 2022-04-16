#include <iostream>
#include <exercise2/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 2
    ExerciseImpl("Exercise 2 : block effet with less threads per block")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

#include <iostream>
#include <exercise5/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    findCudaDevice(argc, argv);
    // run exercise 5
    ExerciseImpl("Exercise 5 : filtrage d'image")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

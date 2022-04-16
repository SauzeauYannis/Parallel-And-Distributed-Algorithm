#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise1/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    // run exercise 3
    ExerciseImpl("Exercise 1 : partition")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

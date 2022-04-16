#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise4/ExerciseImpl.h>


int main(int argc, const char**argv) 
{
    // run exercise 4
    ExerciseImpl("Exercise 4")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise3/ExerciseImpl.h>


int main(int argc, const char**argv) 
{
    // run exercise 3
    ExerciseImpl("Exercise 3")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise5/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    // run exercise 3
    ExerciseImpl("Exercise 5 : parallel radix sort")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

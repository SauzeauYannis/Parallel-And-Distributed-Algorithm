#include <iostream>
#include <exercise2/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    // run exercise 2
    ExerciseImpl("Exercise 2: histogram calculation")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

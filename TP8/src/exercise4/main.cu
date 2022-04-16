#include <iostream>
#include <exercise4/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    // run exercise 4
    ExerciseImpl("Exercise 4: calculate the final transformation")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

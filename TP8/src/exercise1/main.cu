#include <iostream>
#include <exercise1/ExerciseImpl.h>

int main(int argc, const char**argv) 
{
    // run exercise 1
    ExerciseImpl("Exercise 1: RGB->HSV->RGB color space transformation")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

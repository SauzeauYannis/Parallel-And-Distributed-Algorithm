#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise1/Exercise1.h>


int main(int argc, const char**argv) 
{
    // run exercise 1
    Exercise1("Exercise 1")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

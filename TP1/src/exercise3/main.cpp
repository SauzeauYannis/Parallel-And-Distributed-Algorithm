#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise3/Exercise3.h>


int main(int argc, const char**argv) 
{
    // run exercise 3
    Exercise3("Exercise 3")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

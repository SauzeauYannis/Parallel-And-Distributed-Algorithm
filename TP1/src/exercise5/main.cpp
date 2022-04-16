#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise5/Exercise5.h>


int main(int argc, const char**argv) 
{
    // run exercise 5
    Exercise5("Exercise 5")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

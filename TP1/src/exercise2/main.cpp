#pragma warning( disable : 4244 ) 

#include <iostream>

#include <exercise2/Exercise2.h>


int main(int argc, const char**argv) 
{
    // run exercise 2
    Exercise2("Exercise 2")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}

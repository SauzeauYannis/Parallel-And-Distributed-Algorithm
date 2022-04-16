#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "iMatrix22.h"

std::ostream& operator<<(std::ostream& os, iMatrix22& m) 
{
    os << "iMatrix22{"
        << m.values[0] << "," 
        << m.values[1] << "," 
        << m.values[2] << "," 
        << m.values[3] << "}";
    return os;
}
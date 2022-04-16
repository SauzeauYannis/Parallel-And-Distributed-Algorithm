#pragma once

#include <Exercise.h>
#include <exo1/student.h>

class Exercise1 : public Exercise 
{
public:
    Exercise1(const std::string& name ) 
        : Exercise(name, new StudentWork1())
    {}

    Exercise1() = delete;
    Exercise1(const Exercise1&) = default;
    ~Exercise1() = default;
    Exercise1& operator= (const Exercise1&) = delete;

    Exercise1& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    
};
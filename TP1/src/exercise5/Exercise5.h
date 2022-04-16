#pragma once

#include <utility> // std::pair
#include <Exercise.h>
#include <exo5/student.h>

class Exercise5 : public Exercise 
{
public:
    Exercise5(const std::string& name ) 
        : Exercise(name, new StudentWork5())
    {}

    Exercise5() = delete;
    Exercise5(const Exercise5&) = default;
    ~Exercise5() = default;
    Exercise5& operator= (const Exercise5&) = delete;

    Exercise5& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    

    unsigned number_of_threads = 4;
    unsigned interval_start=2;
    unsigned interval_end=1000; // interval
    std::vector<std::pair<unsigned,unsigned>> twin_primes;
};
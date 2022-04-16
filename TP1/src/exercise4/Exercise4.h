#pragma once

#include <Exercise.h>
#include <exo4/student.h>

class Exercise4 : public Exercise 
{
public:
    Exercise4(const std::string& name ) 
        : Exercise(name, new StudentWork4())
    {}

    Exercise4() = delete;
    Exercise4(const Exercise4&) = default;
    ~Exercise4() = default;
    Exercise4& operator= (const Exercise4&) = delete;

    Exercise4& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int); 

    double pi_student = 0.0;
    unsigned number_of_threads = 4;
};
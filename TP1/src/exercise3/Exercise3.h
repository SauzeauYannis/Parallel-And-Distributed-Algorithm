#pragma once

#include <Exercise.h>
#include <exo3/student.h>

class Exercise3 : public Exercise 
{
public:
    Exercise3(const std::string& name ) 
        : Exercise(name, new StudentWork3()), pi_student(0.0), number_of_threads(8)
    {}

    Exercise3() = delete;
    Exercise3(const Exercise3&) = default;
    ~Exercise3() = default;
    Exercise3& operator= (const Exercise3&) = delete;

    Exercise3& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    

    double pi_student = 0.0;
    unsigned number_of_threads = 4;
};
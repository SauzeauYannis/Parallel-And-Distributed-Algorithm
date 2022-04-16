#pragma once

#include <Exercise.h>
#include <exo2/student.h>

class Exercise2 : public Exercise 
{
public:
    Exercise2(const std::string& name ) 
        : Exercise(name, new StudentWork2()), pi_student(0.0), number_of_threads(8)
    {}

    Exercise2() = delete;
    Exercise2(const Exercise2&) = default;
    ~Exercise2() = default;
    Exercise2& operator= (const Exercise2&) = delete;

    Exercise2& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);    

    double pi_student = 0.0;
    unsigned number_of_threads = 4;
};
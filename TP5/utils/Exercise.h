#pragma once

#include <iostream>
#include <string>
#include <helper_string.h>
#include <chronoCPU.hpp>
//#include <chronoGPU.hpp>
#include <StudentWork.h>

class Exercise 
{
public:

    const std::string name;

    Exercise(const std::string& name, StudentWork*student=nullptr) 
        : name(name), student(student)
    {}

    Exercise() = delete;
    Exercise(const Exercise&) = default;
    ~Exercise() = default;
    Exercise& operator= (const Exercise&) = default;
    
    void evaluate(const bool verbose=true) {
        if( !verifyConfiguration(verbose) )
            return ;
        if (verbose) 
            std::cout << "Run exercise " << name << "..." << std::endl;
        run(verbose);
        if ( check() ) 
            std::cout << "Well done: exercise " << name << " SEEMS TO WORK!" << std::endl;
        else
            std::cout << "Bad job: exercise " << name << " DOES NOT WORK!" << std::endl;
    }

    virtual Exercise& parseCommandLine(const int argc, const char**argv) {
        return *this;
    }

protected:
    StudentWork*student;

    void setStudentWork(StudentWork*const student) 
    {
        this->student = student;
    }

    bool verifyConfiguration(const bool verbose) const {
        if (student == nullptr) {
            std::cerr << "Exercise " << name << " not configurated correctly!" << std::endl;
            return false;
        }
        if ( !(student->isImplemented()) ) {
            std::cout << "Exercise " << name << " not implement yet..." << std::endl;
            return false;
        }
        return true;
    }

    virtual void run(const bool verbose) = 0;
    virtual bool check()  = 0;
    
    int getNFromCmdLine(const int argc, const char**argv, int N = 8, const int maxN = 29 ) const
    {
        if( checkCmdLineFlag(argc, argv, "n") ) {
            int value;
            getCmdLineArgumentValue(argc, argv, "n", &value);
            std::cout << "\tfind command line parameter -n=" << value << std::endl;
            if( value >= 1 && value <= maxN )
                N = value;
            else
                std::cerr << "\tWarning: parameter must be positive and lesser than " << maxN << std::endl;
        }
        return N;
    }
    
    template<typename Fun>
    void execute_and_display_time(const bool verbose, const Fun&fun, const unsigned nb_try=20) 
    {
         ChronoCPU chr;
        long long minTime = 0x7fffffffffffffff; // 64 bits, donc 8 octets, soit 16 lettres (4 bits par lettre)
        for(unsigned i=0; i<nb_try; ++i)
        {
            chr.start();
            {
                fun();
            }
            chr.stop();
            auto us = chr.elapsedTimeInMicroSeconds();
            if( us < minTime ) minTime = us;
        }
        if( verbose ) {
            display_time(minTime);
        }
    }

    void display_time(const long long us) const
    {
        std::cout << "\tStudent's Work Done in ";
        if (us < 100*1000)  
            std::cout << us << " us";
        else if ( us < 100*1000*1000 ) 
            std::cout << us/1000 << " ms";
        else
            std::cout << us/(1000*1000) << " sec.";
        std::cout << std::endl;
    }
};
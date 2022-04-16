#pragma once

#include <Exercise.h>
#include <chronoGPU.hpp>

class ExerciseGPU : public Exercise 
{
public:
    ExerciseGPU(const std::string& name, StudentWork*student=nullptr) 
        : Exercise(name, student)
    {}

    ExerciseGPU() = delete;
    ExerciseGPU(const ExerciseGPU&) = default;
    ~ExerciseGPU() = default;
    ExerciseGPU& operator= (const ExerciseGPU&) = default;

    template<typename Fun>
    void execute_and_display_GPU_time(const bool verbose, const Fun&fun, const unsigned nb_try=20) 
    {
        ChronoGPU chr;
        chr.start();
        {
            for(unsigned i=0; i<nb_try; ++i)
                fun();
        }
        chr.stop();
        if( verbose ) {
            auto us = chr.elapsedTime() * 1000.f; // wait us, have ms
            us /= static_cast<decltype(us)>(nb_try);
            std::cout << "\tStudent's Work Done in ";        
            display_time(static_cast<long long>(us));
            std::cout << std::endl;
        }
    }
};
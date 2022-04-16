#pragma once

#include <StudentWork.h>

class StudentWork2: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWork2() = default; 
	StudentWork2(const StudentWork2&) = default;
	~StudentWork2() = default;
	StudentWork2& operator=(const StudentWork2&) = default;

	double run(const unsigned nb_threads);
};
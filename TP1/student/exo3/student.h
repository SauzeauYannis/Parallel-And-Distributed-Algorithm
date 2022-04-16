#pragma once

#include <StudentWork.h>

class StudentWork3: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWork3() = default; 
	StudentWork3(const StudentWork3&) = default;
	~StudentWork3() = default;
	StudentWork3& operator=(const StudentWork3&) = default;

	double run(const unsigned nb_threads);
};
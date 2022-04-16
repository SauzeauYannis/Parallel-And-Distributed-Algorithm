#pragma once

#include <StudentWork.h>

class StudentWork4: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWork4() = default; 
	StudentWork4(const StudentWork4&) = default;
	~StudentWork4() = default;
	StudentWork4& operator=(const StudentWork4&) = default;

	double run(const unsigned nb_threads);
};
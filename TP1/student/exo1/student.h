#pragma once

#include <StudentWork.h>

class StudentWork1 : public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWork1() = default; 
	StudentWork1(const StudentWork1&) = default;
	~StudentWork1() = default;
	StudentWork1& operator=(const StudentWork1&) = default;

	void run();
};
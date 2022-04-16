#pragma once

#include <StudentWork.h>
#include <vector>

class StudentWorkImpl: public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	long long run_reduce(
		const std::vector<long long>& input
	);
};
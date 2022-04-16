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

	void run_square(
		const std::vector<int>& input,
		std::vector<int>& output
	);
	
	void run_sum(
		const std::vector<int>& input_a,
		const std::vector<int>& input_b,
		std::vector<int>& output
	);
};
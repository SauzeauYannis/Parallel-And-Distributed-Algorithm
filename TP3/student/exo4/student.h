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

	void run_gather(
		const std::vector<int>& input,
		const std::vector<size_t>& map,
		std::vector<int>& output
	);
	
	void run_scatter(
		const std::vector<int>& input,
		const std::vector<size_t>& map,
		std::vector<int>& output
	);
};
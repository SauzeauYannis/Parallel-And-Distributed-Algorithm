#pragma once

#include <StudentWork.h>
#include <vector>
#include <utility>

class StudentWork5 : public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWork5() = default; 
	StudentWork5(const StudentWork5&) = default;
	~StudentWork5() = default;
	StudentWork5& operator=(const StudentWork5&) = default;

	// computes the twin primes into a given interval
	std::vector<std::pair<uint32_t,uint32_t>> run(const unsigned min, const unsigned max, const unsigned nb_threads);
};
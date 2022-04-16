#pragma once

#include <StudentWork.h>
#include <immintrin.h>
#include <vector>

class StudentWorkImpl : public StudentWork
{
public:

	bool isImplemented() const ;

	StudentWorkImpl() = default; 
	StudentWorkImpl(const StudentWorkImpl&) = default;
	~StudentWorkImpl() = default;
	StudentWorkImpl& operator=(const StudentWorkImpl&) = default;

	float run(
		__m256 const * const input, 
		const size_t size
	);
	float run(
		float const * const input, 
		const size_t size
	);
};
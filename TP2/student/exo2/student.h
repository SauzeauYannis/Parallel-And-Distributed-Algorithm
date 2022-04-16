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

	void run(
		__m256 const * const input, 
		__m256       * const output, 
		const size_t size
	);
	void run(
		float const * const input, 
		float       * const output, 
		const size_t size
	);
};
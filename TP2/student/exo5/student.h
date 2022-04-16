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
		__m256 const * const input_matrix, // square matrix, stored by row
		__m256 const * const input_vector, 
		float        * const output_vector, 
		const size_t vector_size
	);
	void run(
		float const * const input_matrix, // square matrix, stored by row
		float const * const input_vector, 
		float       * const output_vector, 
		const size_t vector_size
	);
};
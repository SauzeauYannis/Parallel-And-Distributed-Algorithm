#include <immintrin.h> //AVX+SSE Extensions#include <vector>
#include <cmath>
#include <iostream>
#include <exo2/student.h>

namespace {

	struct Convertor 
	{ 
		union {
			__m256 avx; 
			float f[8];
		} u;
		// constructor
		Convertor(const __m256& m256) { u.avx = m256; };
		// accessor to element i (between 0 and 7 included)
		float operator()(int i) const 
		{ 			
			return u.f[i]; 
		}
		// prints data on a given stream
		friend std::ostream& operator<<(std::ostream&, const Convertor&c);
	};

	std::ostream& operator<<(std::ostream& os, const Convertor&c) 
	{
		os << "{ ";
		for(int i=0; i<7; ++i) {
			os << c(i) << ", ";
		}
		return os << c(7) << " }";
	}
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

/// calculate with floats
/// @param input : array containing float values
/// @param output : OUTPUT array containing the square roots of the input (to compute)
/// @param size : size of the arrays "input" and "output", in number of floats
#pragma optimize("", off)
void StudentWorkImpl::run(float const * const input, float * const output, const size_t size) {
	for (int i = 0; i < size; ++i)
		output[i] = sqrt(input[i]);
}

/// calculate with mm256 (8 floats)
/// @param input : array containing AVX values
/// @param output : OUTPUT array containing the square roots of the input (to compute)
/// @param size : size of the arrays "input" and "output", in number of AVX values
#pragma optimize("", off)
void StudentWorkImpl::run(__m256 const *const input, __m256 * const output, const size_t size) {
	for (int i = 0; i < size; ++i)
		output[i] = _mm256_sqrt_ps(input[i]);
}
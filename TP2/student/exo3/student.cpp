#include <immintrin.h> //AVX+SSE Extensions#include <vector>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm> 
#include <exo3/student.h>

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

	// partial calculation for linear form
#pragma optimize("", off)
	void calculate_linear(float const * const input, float * const output, const size_t first, const size_t last)
	{
		for (int i = first; i < last; ++i) 
			output[i] = sqrt(input[i]);
	}

	// partial calculation for AVX vectorial form
#pragma optimize("", off)	
	void calculate_vectorial(__m256 const *const input, __m256 * const output, const size_t first, const size_t last)
	{
		for (int i = first; i < last; ++i) 
			output[i] = _mm256_sqrt_ps(input[i]);
	}

}

bool StudentWorkImpl::isImplemented() const {
	return true;
}


// partial calculation for linear form
#pragma optimize("", off)
void StudentWorkImpl::run(const size_t nb_threads, float const * const input, float * const output, const size_t size) 
{
	// travail constant par threads (sqrt est une instruction) ... on divise en le nombre de threads
	const size_t packet_size = (size + nb_threads-1) / nb_threads;
	std::vector<std::thread> threads(nb_threads);

	for (int i = 0; i < nb_threads; ++i) {
		size_t last = (i + 1) * packet_size;
		if (last > size) last = size;
		threads[i] = std::thread(calculate_linear, input, output, i * packet_size, last);
	}

	for (int i = 0; i < nb_threads; ++i)
		threads[i].join();
}

// partial calculation for vectorial form
#pragma optimize("", off)
void StudentWorkImpl::run(const size_t nb_threads, __m256 const *const input, __m256 * const output, const size_t size) 
{
	// travail constant par threads (sqrt est une instruction) ... on divise en le nombre de threads
	const size_t packet_size = (size + nb_threads-1) / nb_threads;
	std::vector<std::thread> threads(nb_threads);

	for (int i = 0; i < nb_threads; ++i) {
		size_t last = (i + 1) * packet_size;
		if (last > size) last = size;
		threads[i] = std::thread(calculate_vectorial, input, output, i * packet_size, last);
	}
		
	for (int i = 0; i < nb_threads; ++i)
		threads[i].join();
}

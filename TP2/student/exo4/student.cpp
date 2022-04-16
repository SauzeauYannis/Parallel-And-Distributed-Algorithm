#include <immintrin.h> //AVX+SSE Extensions#include <vector>
#include <cmath>
#include <iostream>
#include <exo4/student.h>

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
		float& operator()(int i) 
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

// calculate with floats
#pragma optimize("", off)
float StudentWorkImpl::run(float const * const input, const size_t size) 
{
	if (size == 0) return 0.0;

	size_t i = size - 1;
	float sum = input[i];

	while (i-- > 0) 
		sum += input[i];

	return sum;
}

// calculate with mm256
#pragma optimize("", off)
float StudentWorkImpl::run(__m256 const *const input, const size_t size) 
{
	if (size == 0) return 0.0;

	size_t i = size - 1;
	__m256 sum = input[i];

	while (i-- > 0) 
		sum = _mm256_add_ps(sum, input[i]);

	/*
		Ici on fait le produit vectoriel entre notre AVX qui contient
		les sommes et un AVX rempli de 1, puis on met un masque égal
		à 241 qui vaut en binaire 1111 0001 ce qui va au final stocker
		une partie de la somme à la case 0 de notre AVX et l'autre
		partie de la somme à la case 4.
	*/
	__m256 dp = _mm256_dp_ps(sum, _mm256_set1_ps(1.0), 241);

	Convertor c = Convertor(dp);

	return c(0) + c (4);
}

#include <immintrin.h> //AVX+SSE Extensions

#include <iostream>
#include <exo1/student.h>

namespace {
	// tricky trick (enfin je sais pas trop comment cela s'écrit mais l'idée est là)
	// NB: this is unnecessary both on Linux/Windows, but they provide different mechanism to
	//     access the individual values ... So we need this for general access. 
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

// see documentation for _mm256_load_ps, _mm256_add_ps and _mm256_sqrt_ps!
void StudentWorkImpl::run() 
{
	float values[8] = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};

	__m256 c = 
		_mm256_add_ps( 
			_mm256_load_ps(values), 
			_mm256_load_ps(values)
		);

	std::cout << "additions are " << Convertor(c) << std::endl;

	std::cout << "and square root are " << Convertor(_mm256_sqrt_ps(c)) << std::endl;
}

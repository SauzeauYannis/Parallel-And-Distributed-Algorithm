#include <iostream>
#include <exo1/student.h>
#include <exo1/BinaryMap.h>
#include <OPP_cuda.cuh>


// NB: tout le travail à effectuer l'est dans le fichier BinaryMap.h !
namespace 
{
	// do not modify this functor ;-)
	// but notice the __device__ keyword!
	template<typename T>
	struct Plus 
	{
		__device__
		T operator()(const T& a, const T&b) const 
		{
			return a + b;
		}
	};

}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

// Attention : ici la taille des vecteurs n'est pas toujours un multiple du nombre de threads !
// Il faut donc corriger l'exemple du cours ...
void StudentWorkImpl::run_binary_map(
	OPP::CUDA::DeviceBuffer<int>& dev_a,
	OPP::CUDA::DeviceBuffer<int>& dev_b,
	OPP::CUDA::DeviceBuffer<int>& dev_result
) {
	::BinaryMap<int,::Plus<int>>(
		dev_a, dev_b, dev_result, ::Plus<int>()
	);
}

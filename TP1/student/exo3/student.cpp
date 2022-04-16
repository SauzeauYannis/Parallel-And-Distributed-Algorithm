#include <thread> // C++ 11
#include <mutex> // C++ 11
#include <iostream>
#include <cmath>
#include <vector>
#include <exo3/student.h>

namespace {

	std::mutex mutex;
	double result = 0.0;
	const unsigned int limit = 1 << 28; // 2^28 == 256 millions

	void calculate(int num, const unsigned nb_threads) {
		double acc = 0.0;
		for (unsigned int n = num; n < limit; n += nb_threads)
			acc += pow(-1.0, n) / (2.0 * n + 1.0);
		mutex.lock();   // Enter the critical section
		result += acc;  // Write protected variable
		mutex.unlock(); // Exit the critical section
	}

}

bool StudentWork3::isImplemented() const {
	return true;
}

/// nb_threads is between 1 to 64 ...
double StudentWork3::run(const unsigned nb_threads) {
	std::cout << "starting " << nb_threads << " threads ..." << std::endl;
	std::thread *threads = new std::thread[nb_threads];

	for (int i = 0; i < nb_threads; ++i)
		threads[i] = std::thread(calculate, i, nb_threads);

	// synchronize threads:
	for (int i = 0; i < nb_threads; ++i)
		threads[i].join();

	std::cout << "threads have completed." << std::endl;
	double pi = result * 4.0;

	std::cout.precision(12);
	std::cout << "our evaluation is: " << std::fixed << pi << std::endl;

	delete[] threads;

	return pi;
}

#include <thread> // C++ 11
#include <mutex> // C++ 11
#include <iostream>
#include <cmath>
#include <vector>
#include <exo4/student.h>

namespace {

	const unsigned int limit = 1 << 28; // 2^28 == 256 millions

	class Monitor 
	{
		public:
			Monitor() 
			: mutex{new std::mutex()}, result(0.0) 
			{}

			~Monitor() { delete mutex; }

			void addToResult(double value) {
				mutex->lock();
				result += value;
				mutex->unlock();
			}

			double getResult() const { return result; }

		private:
			std::mutex *mutex;
			double result;
	};

}

bool StudentWork4::isImplemented() const {
	return true;
}

/// nb_threads is between 1 to 64 ...
double StudentWork4::run(const unsigned nb_threads) {
	std::cout << "starting " << nb_threads << " threads ..." << std::endl;
	std::thread *threads = new std::thread[nb_threads];
	Monitor monitor;

	for (int i = 0; i < nb_threads; ++i) {
		threads[i] = std::thread([i, nb_threads, &monitor] {
			double acc = 0.0;
			for (unsigned int n = i; n < limit; n += nb_threads)
				acc += pow(-1.0, n) / (2.0 * n + 1.0);
			monitor.addToResult(acc);
		});
	}

	// synchronize threads:
	for (int i = 0; i < nb_threads; ++i)
		threads[i].join();

	std::cout << "threads have completed." << std::endl;
	double pi = monitor.getResult() * 4.0;

	std::cout.precision(12);
	std::cout << "our evaluation is: " << std::fixed << pi << std::endl;

	delete[] threads;

	return pi;
}

#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <iostream>
#include "student.h"

namespace 
{

	class PrimeTwinsMonitor 
	{
		public:
			PrimeTwinsMonitor()
			: mutex{new std::mutex()}, primes() 
			{}

			~PrimeTwinsMonitor() { delete mutex; }

			void addPrime(std::pair<uint32_t, uint32_t> prime) {
				mutex->lock();
				primes.push_back(prime);
				mutex->unlock();
			}
						
			std::vector<std::pair<uint32_t, uint32_t>> getPrimes() const {
				return primes;
			}

		private:
			std::mutex *mutex;
			std::vector<std::pair<uint32_t, uint32_t>> primes;
	};

	class IntervalMonitor 
	{
		public:
			IntervalMonitor(int min, int max) 
			: mutex{new std::mutex()}, val(min), max(max) 
			{}

			~IntervalMonitor() { delete mutex; }

			int getNumber() {
				int result;
				mutex->lock();
 				result = val;
				val++;
				mutex->unlock();
				return result;
			}

		private:
			std::mutex *mutex;
			int val;
			int max;
	};

	bool are_2_pairs_sorted(const std::pair<uint32_t,uint32_t>& a, const std::pair<uint32_t,uint32_t>& b) {
		return std::get<0>(a) < std::get<0>(b);
	}

	bool is_prime(const uint32_t n) {
		// check division from 2 to n (not efficient at all!)
		for (uint32_t d = 2; d < n; ++d)
			if ((n % d) == 0 ) // d is a divisor, n is not prime
				return false;
		// we have not found any divisor: n is prime
		return true;
	}

}

bool StudentWork5::isImplemented() const {
	return true;
}

std::vector<std::pair<uint32_t,uint32_t>> 
StudentWork5::run(const uint32_t min, const uint32_t max, const uint32_t nb_threads) 
{
	std::vector<std::pair<uint32_t, uint32_t>> result;
	PrimeTwinsMonitor prime_monitor;
	IntervalMonitor interval_monitor(min, max);

	std::cout << "starting " << nb_threads << " threads ..." << std::endl;
	std::thread *threads = new std::thread[nb_threads];

	for (int i = 0; i < nb_threads; ++i) {
		threads[i] = std::thread([max, &interval_monitor, &prime_monitor] {
			int num = interval_monitor.getNumber();
			while (num <= max) {
				if (is_prime(num) && is_prime(num + 2)) 
					prime_monitor.addPrime({num, num + 2});

				num = interval_monitor.getNumber();
			}
		});
	}

	// synchronize threads:
	for (int i = 0; i < nb_threads; ++i)
		threads[i].join();

	result = prime_monitor.getPrimes();

	std::cout << "threads have completed." << std::endl;

	std::sort(result.begin(), result.end(), are_2_pairs_sorted);

	delete[] threads;

	return result;
}

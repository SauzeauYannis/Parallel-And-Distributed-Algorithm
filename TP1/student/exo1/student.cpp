#include <thread>
#include <iostream>
#include <exo1/student.h>

namespace {
	
	void foo() {
		std::cout << "I'm doing foo things ..." << std::endl;
	}

	void bar(int x) {
		std::cout << "I'm doing bar(" << x << ") things ..." << 
		std::endl;
	}

}

bool StudentWork1::isImplemented() const {
	return true;
}

void StudentWork1::run() {
	std::cout << "main thread is launching foo and bar threads ..." <<
	std::endl;

	std::thread first (foo);     // spawn new thread that calls foo()
	std::thread second (bar, 0); // spawn new thread that calls bar(0)

	std::cout << "main, foo and bar now execute concurrently...\n";

	// synchronize threads
	first.join();   // pauses until first finishes
	second.join();  // pauses until second finishes

	std::cout << "foo and bar completed.\n";
}

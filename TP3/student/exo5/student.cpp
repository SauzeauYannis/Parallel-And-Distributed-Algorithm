#include <iostream>
#include <exo5/student.h>

#include <exo5/reduce.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

long long StudentWorkImpl::run_reduce(
	const std::vector<long long>& input
) {
	return OPP::reduce(
		input.begin(), input.end(), 
		0ll, 
		[](const long long& a, const long long& b) { return a + b; } 
	);
}

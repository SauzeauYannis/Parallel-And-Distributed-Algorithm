#include <iostream>
#include <exo4/student.h>

#include <exo4/gather.h>
#include <exo4/scatter.h>

namespace {
}

bool StudentWorkImpl::isImplemented() const {
	return true;
}

void StudentWorkImpl::run_gather(
	const std::vector<int>& input,
	const std::vector<size_t>& map,
	std::vector<int>& output
) {
	OPP::gather(input.begin(), input.end(), map.begin(), output.begin());
}

void StudentWorkImpl::run_scatter(
	const std::vector<int>& input,
	const std::vector<size_t>& map,
	std::vector<int>& output
) {
	OPP::scatter(input.begin(), input.end(), map.begin(), output.begin());
}

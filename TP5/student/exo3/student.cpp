#include <OPP.h>
#include <exo3/student.h>

#include <iostream>

namespace {}

bool StudentWorkImpl::isImplemented() const { return true; }

void StudentWorkImpl::check() {
  std::vector<long> vec(16);

  OPP::transform(OPP::CountingIterator(0), OPP::CountingIterator(16),
                 vec.begin(),
                 std::function([](int a) -> long { return long(a); }));

  std::cout << "check counting iterator :" << std::endl << " --> ";
  for (auto v : vec) std::cout << v << " ";
  std::cout << std::endl;

  auto transformIterator = OPP::make_transform_iterator(
      OPP::CountingIterator(0l),
      std::function([](long a) -> int { return 2 * a + 980; }));

  OPP::transform(transformIterator + 0, transformIterator + 16, vec.begin(),
                 std::function([](int a) -> long { return long(a); }));

  std::cout << "check transform iterator :" << std::endl << " --> ";
  for (auto v : vec) std::cout << v << " ";
  std::cout << std::endl;
}
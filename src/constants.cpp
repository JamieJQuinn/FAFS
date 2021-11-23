#include <iostream>

#include <constants.hpp>

Constants::Constants():
  nx{256},
  ny{256},
  ng{1},
  dt{0.001},
  totalTime{0.7},
  Re{100},
  isAdvectionImplicit{false},
  isDiffusionImplicit{false}
{
  dx = 1.0/(nx+1);
  dy = 1.0/(ny+1);
}

void Constants::print() const {
  std::cout << "nx: " << nx << std::endl;
  std::cout << "ny: " << ny << std::endl;
  std::cout << "ng: " << ng << std::endl;
  std::cout << "dt: " << dt << std::endl;
  std::cout << "totalTime: " << totalTime << std::endl;
  std::cout << "Re: " << Re << std::endl;
  std::cout << "nu: " << 1.0f/Re << std::endl;
  std::cout << "Numerical nu: " << dx*dy/dt << std::endl;
}

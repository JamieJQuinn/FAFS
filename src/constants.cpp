#include <iostream>

#include <constants.hpp>

Constants::Constants():
  nx{128},
  ny{nx},
  ng{1},
  dt{0.01},
  totalTime{1},
  Re{100},
  isAdvectionImplicit{true},
  isDiffusionImplicit{true}
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

#include "array2d.hpp"

int calcTotalArraySize(const int nx, const int ny, const int ng) {
  return (nx+2*ng)*(ny+2*ng);
}

array makeArray(const int nx, const int ny, const int ng) {
  return std::make_unique<real[]>(calcTotalArraySize(nx, ny, ng));
}

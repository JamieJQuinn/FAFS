#include <variables.hpp>
#include <array2d.hpp>

Variables::Variables(const int nx, const int ny, const int ng):
  vx{makeArray(nx, ny, ng)}
{}

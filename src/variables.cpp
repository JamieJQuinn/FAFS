#include <variables.hpp>
#include <constants.hpp>
#include <array2d.hpp>

Variables::Variables(const Constants& c):
  vx(c, "vx"),
  vy(c, "vy"),
  p(c, "pressure")
{}

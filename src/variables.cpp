#include <variables.hpp>
#include <constants.hpp>
#include <array2d.hpp>

Variables::Variables(const Constants& c):
  vx(c.nx, c.ny, c.ng, "vx"),
  vy(c.nx, c.ny, c.ng, "vy"),
  p(c.nx+1, c.ny+1, c.ng, "pressure")
{}

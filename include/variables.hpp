#pragma once

#include <memory>

#include <precision.hpp>
#include <array2d.hpp>

template <class T>
class Variables {
  public:
    T vx, vy, p;

  Variables<T>(const Constants& c):
    vx(c.nx, c.ny, c.ng, "vx"),
    vy(c.nx, c.ny, c.ng, "vy"),
    p(c.nx+1, c.ny+1, c.ng, "pressure")
  {}
};

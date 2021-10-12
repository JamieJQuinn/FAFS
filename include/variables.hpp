#pragma once

#include <memory>

#include <precision.hpp>
#include <array2d.hpp>

class Variables {
  public:
    Array vx, vy, p;
    Variables(const Constants& c);
};

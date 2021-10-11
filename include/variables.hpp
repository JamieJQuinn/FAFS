#pragma once

#include <memory>
#include "precision.hpp"

class Variables {
  public:
    std::unique_ptr<real[]> vx;
    Variables(const int nx, const int ny, const int ng);
};

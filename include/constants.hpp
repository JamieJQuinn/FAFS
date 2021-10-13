#pragma once

#include <precision.hpp>

class Constants {
  public:
    Constants();

    int nx;
    int ny;
    int ng;
    real dx;
    real dy;

    real dt;
    real totalTime;

    real nu; // viscosity
};

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

    real Re; // Reynolds number (in this non-dimensionalisation, equiv to 1/viscosity)
};

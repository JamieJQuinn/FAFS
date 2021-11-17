#pragma once

#include <array2d.hpp>
#include <precision.hpp>

template<class T>
T clamp(const T i, const T upper, const T lower) {
  return std::max(std::min(i, upper), lower);
}

real calcAdvection(const Array& f, const int i, const int j, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng, const Array& vx, const Array& vy);
real calcJacobiStep(const Array& f, const real alpha, const real beta, const Array& b, const int i, const int j);
void applyJacobiStep(Array& out, const Array& f, const real alpha, const real beta, const Array& b);
void runJacobiIteration(Array& out, Array& in, const real alpha, const real beta, const Array& b, const int iterations=20);
real ddx(const Array& f, const real dx, const int i, const int j);
real ddy(const Array& f, const real dy, const int i, const int j);
void advectImplicit(Array& out, const Array& f, const Array& vx, const Array& vy, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng);
void calcAdvectionTerm(Array& out, const Array& f, const Array& vx, const Array& vy, const real dx, const real dy);
void calcDiffusionTerm(Array& out, const Array& f, const real dx, const real dy);
void advanceEuler(Array& out, const Array& ddt, const real dt);
void calcDivergence(Array& out, const Array& fx, const Array& fy, const real dx, const real dy);
void applyProjectionX(Array& out, const Array& f, const real dx);
void applyProjectionY(Array& out, const Array& f, const real dy);

#include <iostream>
#include <memory>

#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>
#include <hdffile.hpp>

void setInitialConditions(Variables& vars, const Constants& c) {
  //for (int i=c.nx/4; i<3*c.nx/4; ++i) {
  for (int i=0; i<c.nx/2; ++i) {
    //for (int j=c.ny/4; j<3*c.ny/4; ++j) {
    for (int j=0; j<c.ny/2; ++j) {
      vars.vx(i,j) = 9.0;
    }
  }
}

void applyBoundaryConditions(Variables& vars, const Constants& c) {
  for (int i=0; i<c.nx; ++i) {
    vars.vx(i, -1) = -vars.vx(i, 0);
    vars.vx(i, c.ny) = -vars.vx(i, c.ny-1);
  }
  for(int j=0; j<c.ny; ++j) {
    vars.vx(-1, j) = -vars.vx(0, j);
    vars.vx(c.nx, j) = -vars.vx(c.nx-1, j);
  }
}

void render(const Variables& vars) {
  vars.vx.render();
}

real calcDiffusion(const Array& f, const int i, const int j, const real dx, const real dy) {
  return (f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j) - 4*f(i,j))/(dx*dy);
}

real identityKernel(const Array& f, const int i, const int j) {
  return f(i,j);
}

int main() {
  const Constants c;

  Variables vars(c);

  setInitialConditions(vars, c);

  Array out(c, "temp");

  real t=0;
  real total_time = 0.01;
  real dt = 0.001;

  auto diffusionKernel = [&](const Array& f, const int i, const int j) {
    return calcDiffusion(f, i, j, c.dx, c.dy);
  };

  auto updateKernel = [&](Array& f, const int i, const int j) {
    f(i,j) += out(i,j)*dt;
  };

  HDFFile icFile("000000.hdf5");
  vars.vx.saveTo(icFile.file);
  icFile.close();

  while (t < total_time) {
    vars.vx.applyKernel(diffusionKernel, out);
    vars.vx.applyKernel(updateKernel);
    applyBoundaryConditions(vars, c);

    t += dt;
  }

  vars.vx.render();

  HDFFile laterFile("000001.hdf5");
  vars.vx.saveTo(laterFile.file);
  icFile.close();
}

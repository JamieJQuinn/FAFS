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
      vars.vx(i,j) = 0.1;
    }
  }
}

void applyBoundaryConditions(Variables& vars, const Constants& c) {
  for (int i=0; i<c.nx; ++i) {
    vars.vx(i, -1) = -vars.vx(i, 0);
    vars.vx(i, c.ny) = -vars.vx(i, c.ny-1);
    vars.vy(i, -1) = -vars.vy(i, 0);
    vars.vy(i, c.ny) = -vars.vy(i, c.ny-1);
  }
  for(int j=0; j<c.ny; ++j) {
    vars.vx(-1, j) = -vars.vx(0, j);
    vars.vx(c.nx, j) = -vars.vx(c.nx-1, j);
    vars.vy(-1, j) = -vars.vy(0, j);
    vars.vy(c.nx, j) = -vars.vy(c.nx-1, j);
  }
}

void render(const Variables& vars) {
  vars.vx.render();
}

real calcDiffusion(const Array& f, const int i, const int j, const real nu, const real dx, const real dy) {
  return nu*(f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j) - 4*f(i,j))/(dx*dy);
}

int clamp(const int i, const int upper, const int lower) {
  return std::max(std::min(i, upper), lower);
}

real calcAdvection(const Array& q, const int i, const int j, const real dx, const real dy, const int nx, const int ny, const int ng, const Array& vx, const Array& vy) {
  // figure out where the current piece has come from (in index space)
  real oldi = i - vx(i,j)/dx;
  real oldj = j - vy(i,j)/dy;
  // clamp to int indices and ensure inside domain
  int ip = clamp(int(oldi+1),nx+ng-1,-ng);
  int im = clamp(int(oldi)  ,nx+ng-1,-ng);
  int jp = clamp(int(oldj+1),ny+ng-1,-ng);
  int jm = clamp(int(oldj)  ,ny+ng-1,-ng);
  // bilinearly interpolate
  return (q(ip, jp) + q(ip, jm) + q(im, jp) + q(im, jm))/4;
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
  real total_time = 0.003;
  real dt = 0.001;

  auto diffusionKernel = [&](const Array& f, const int i, const int j) {
    return calcDiffusion(f, i, j, c.nu, c.dx, c.dy);
  };

  auto advectionKernel = [&](const Array& q, const int i, const int j) {
    return calcAdvection(q, i, j, c.dx, c.dy, c.nx, c.ny, c.ng, vars.vx, vars.vy);
  };

  auto updateKernel = [&](Array& f, const int i, const int j) {
    f(i,j) += out(i,j)*dt;
  };

  auto assignKernel = [&](Array& f, const int i, const int j) {
    f(i,j) = out(i,j);
  };

  HDFFile icFile("000000.hdf5");
  vars.vx.saveTo(icFile.file);
  vars.vy.saveTo(icFile.file);
  icFile.close();

  while (t < total_time) {
    applyBoundaryConditions(vars, c);

    vars.vx.applyKernel(advectionKernel, out);
    vars.vx.applyKernel(assignKernel);
    //vars.vy.applyKernel(advectionKernel, out);
    //vars.vy.applyKernel(assignKernel);
    vars.vx.applyKernel(diffusionKernel, out);
    vars.vx.applyKernel(updateKernel);
    //vars.vy.applyKernel(diffusionKernel, out);
    //vars.vy.applyKernel(updateKernel);

    t += dt;
  }

  //vars.vx.render();
  //vars.vy.render();

  HDFFile laterFile("000001.hdf5");
  vars.vx.saveTo(laterFile.file);
  vars.vy.saveTo(laterFile.file);
  out.saveTo(laterFile.file);
  icFile.close();
}

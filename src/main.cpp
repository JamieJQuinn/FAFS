#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>

#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>
#include <hdffile.hpp>

#define DEBUG

void setInitialConditions(Variables& vars) {
  //for (int i=vars.vx.nx/4; i<3*vars.vx.nx/4; ++i) {
    //for (int j=vars.vx.ny/4; j<3*vars.vx.ny/4; ++j) {
      //vars.vx(i,j) = 0.1;
    //}
  //}
  //for (int i=0; i<vars.vx.nx; ++i) {
    //for (int j=0; j<vars.vx.ny; ++j) {
      //real x = i*vars.vx.dx-0.5;
      //real y = j*vars.vx.dy-0.5;
      //real r2 = x*x+y*y;
      //vars.vx(i,j) = 0.01*y/r2;
      //vars.vy(i,j) = -0.01*x/r2;
    //}
  //}
}

void applyNoSlipBC(Array& var) {
  for (int i=0; i<var.nx; ++i) {
    var(i, -1) = 0.0;
    var(i, var.ny) = 0.0;
  }
  for(int j=0; j<var.ny; ++j) {
    var(-1, j) = 0.0;
    var(var.nx, j) = 0.0;
  }
}

void applyVonNeumannBC(Array& var) {
  for (int i=0; i<var.nx; ++i) {
    var(i, -1) = var(i, 0);
    var(i, var.ny) = var(i, var.ny-1);
  }
  for(int j=0; j<var.ny; ++j) {
    var(-1, j) = var(0, j);
    var(var.nx, j) = var(var.nx-1, j);
  }
}

void applyVxBC(Array& vx) {
  applyNoSlipBC(vx);
  // Driven cavity flow
  for (int i=0; i<vx.nx; ++i) {
    vx(i, vx.ny) = 1;
  }
}

void applyVyBC(Array& vy) {
  applyNoSlipBC(vy);
}

void applyBoundaryConditions(Variables& vars) {
  applyVxBC(vars.vx);
  applyVyBC(vars.vy);
}

void render(const Variables& vars) {
  vars.vx.render();
}

int clamp(const int i, const int upper, const int lower) {
  return std::max(std::min(i, upper), lower);
}

real clamp(const real i, const real upper, const real lower) {
  return std::max(std::min(i, upper), lower);
}

real calcAdvection(const Array& f, const int i, const int j, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng, const Array& vx, const Array& vy) {
  // figure out where the current piece has come from (in index space)
  real x = i - dt*vx(i,j)/dx;
  real y = j - dt*vy(i,j)/dy;
  // clamp to int indices and ensure inside domain
  int rigIdx = nx-1+ng;
  int lefIdx = -ng;
  int topIdx = ny-1+ng;
  int botIdx = -ng;
  int x2 = clamp(int(std::floor(x+1)),rigIdx, lefIdx);
  int x1 = clamp(int(std::floor(x))  ,rigIdx, lefIdx);
  int y2 = clamp(int(std::floor(y+1)),topIdx, botIdx);
  int y1 = clamp(int(std::floor(y))  ,topIdx, botIdx);
  x = clamp(x, real(rigIdx), real(lefIdx));
  y = clamp(y, real(topIdx), real(botIdx));
  //std::cout << x << ", " << y << std::endl;
  //std::cout << x1 << ", " << x2 << ", " << y1 << ", " << y2 << std::endl;
  //std::cout << x1 << ", " << int(x) << std::endl;
  //assert(x1 != x2);
  //assert(y1 != y2);
  //// bilinearly interpolate
  real fy1, fy2;
  if(x1!=x2) {
    real x1Weight = (x2-x)/(x2-x1);
    real x2Weight = (x-x1)/(x2-x1);
    fy1 = x1Weight*f(x1, y1) + x2Weight*f(x2, y1);
    fy2 = x1Weight*f(x1, y2) + x2Weight*f(x2, y2);
  } else {
    fy1 = f(x1, y1);
    fy2 = f(x1, y2);
  }
  real fAv;
  if(y1!=y2) {
    real y1Weight = (y2-y)/(y2-y1);
    real y2Weight = (y-y1)/(y2-y1);
    fAv = y1Weight*fy1 + y2Weight*fy2;
  } else {
    fAv = fy1;
  }
  return fAv;
}

real calcJacobiStep(const Array& f, const int i, const int j, const real alpha, const real beta, const Array& b) {
  return (alpha*b(i,j) + f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j))/beta;
}

void runJacobiIteration(Array& in, Array& out, kernelFn fn, const int iterations=20) {
  for(int i=0; i<iterations; ++i) {
    in.applyKernel(fn, out);
    in.swap(out);
  }
}

int main() {
  const Constants c;

  Variables vars(c);

  setInitialConditions(vars);
  applyBoundaryConditions(vars);

  Array boundTemp1(c.nx, c.ny, c.ng, "boundTemp1");
  Array boundTemp2(c.nx, c.ny, c.ng, "boundTemp2");
  Array cellTemp1(c.nx+1, c.ny+1, c.ng, "cellTemp1");
  Array cellTemp2(c.nx+1, c.ny+1, c.ng, "cellTemp2");
  Array divw(c.nx, c.ny, c.ng, "divw");

  auto vxDiffusionJacobiKernel = [&](const Array& f, const int i, const int j) {
    return calcJacobiStep(f, i, j, (c.dx*c.dy)/(c.nu*c.dt), 4+(c.dx*c.dy)/(c.nu*c.dt), vars.vx);
  };

  auto vyDiffusionJacobiKernel = [&](const Array& f, const int i, const int j) {
    return calcJacobiStep(f, i, j, (c.dx*c.dy)/(c.nu*c.dt), 4+(c.dx*c.dy)/(c.nu*c.dt), vars.vy);
  };

  auto projectionJacobiKernel = [&](const Array& f, const int i, const int j) {
    return calcJacobiStep(f, i, j, -c.dx*c.dy, 4.0f, divw);
  };

  auto divergenceKernel = [&](Array& f, const int i, const int j) {
    f(i,j) = (vars.vx(i,j) - vars.vx(i-1,j))/c.dx + (vars.vy(i,j) - vars.vy(i,j-1))/c.dy;
  };

  auto advectionKernel = [&](const Array& q, const int i, const int j) {
    return calcAdvection(q, i, j, c.dx, c.dy, c.dt, c.nx, c.ny, c.ng, vars.vx, vars.vy);
  };

  auto explicitAdvectionKernel = [&](const Array& q, const int i, const int j) {
    return -(vars.vx(i,j) * (q(i+1,j)-q(i-1,j))/(2*c.dx) + vars.vy(i,j) * (q(i,j+1) - q(i,j-1))/(2*c.dy));
  };

  auto eulerKernel = [&](Array& f, const Array& in, const int i, const int j) {
    f(i,j) += in(i,j)*c.dt;
  };

  auto vxProjectKernel = [&](Array& f, const Array& in, const int i, const int j) {
    f(i,j) -= (in(i+1,j)-in(i,j))/c.dx;
  };

  auto vyProjectKernel = [&](Array& f, const Array& in, const int i, const int j) {
    f(i,j) -= (in(i,j+1)-in(i,j))/c.dy;
  };

  HDFFile icFile("000000.hdf5");
  vars.vx.saveTo(icFile.file);
  vars.vy.saveTo(icFile.file);
  icFile.close();

  real t=0;
  while (t < c.totalTime) {
    // ADVECTION
    // implicit
    vars.vx.applyKernel(advectionKernel, boundTemp1);
    vars.vy.applyKernel(advectionKernel, boundTemp2);
    vars.vx.swapData(boundTemp1);
    vars.vy.swapData(boundTemp2);
    // explicit
    //vars.vx.applyKernel(explicitAdvectionKernel, boundTemp1);
    //vars.vy.applyKernel(explicitAdvectionKernel, boundTemp2);
    //vars.vx.applyKernel(eulerKernel, boundTemp1);
    //vars.vy.applyKernel(eulerKernel, boundTemp2);
    applyBoundaryConditions(vars);

    // DIFFUSION
    boundTemp1.initialise(0);
    applyVxBC(boundTemp1);
    applyVxBC(boundTemp2);
    runJacobiIteration(boundTemp1, boundTemp2, vxDiffusionJacobiKernel);
    vars.vx.swapData(boundTemp1);

    boundTemp1.initialise(0);
    applyVyBC(boundTemp1);
    applyVyBC(boundTemp2);
    runJacobiIteration(boundTemp1, boundTemp2, vyDiffusionJacobiKernel);
    vars.vy.swapData(boundTemp1);
    applyBoundaryConditions(vars);

    // PROJECTION
    divw.applyKernel(divergenceKernel);
    cellTemp1.initialise(0);
    runJacobiIteration(cellTemp1, cellTemp2, projectionJacobiKernel);
    vars.p.swapData(cellTemp1);
    applyVonNeumannBC(vars.p);
    vars.vx.applyKernel(vxProjectKernel, vars.p);
    vars.vy.applyKernel(vyProjectKernel, vars.p);
    applyBoundaryConditions(vars);

    t += c.dt;
  }

  HDFFile laterFile("000001.hdf5");
  vars.vx.saveTo(laterFile.file);
  vars.vy.saveTo(laterFile.file);
  vars.p.saveTo(laterFile.file);
  divw.saveTo(laterFile.file);
  icFile.close();
}

#include <iostream>
#include <memory>

#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>
#include <hdffile.hpp>

void setInitialConditions(Variables& vars, const Constants& c) {
  //for (int i=c.nx/4; i<3*c.nx/4; ++i) {
    //for (int j=c.ny/4; j<3*c.ny/4; ++j) {
      //vars.vx(i,j) = 0.1;
    //}
  //}
  //for (int i=0; i<c.nx; ++i) {
    //for (int j=0; j<c.ny; ++j) {
      //real x = i*c.dx-0.5;
      //real y = j*c.dy-0.5;
      //real r2 = x*x+y*y;
      //vars.vx(i,j) = 0.01*y/r2;
      //vars.vy(i,j) = -0.01*x/r2;
    //}
  //}
}

void applyNoSlipBC(Array& var, const Constants& c) {
  for (int i=0; i<c.nx; ++i) {
    var(i, -1) = 0.0;
    var(i, c.ny) = 0.0;
  }
  for(int j=0; j<c.ny; ++j) {
    var(-1, j) = 0.0;
    var(c.nx, j) = 0.0;
  }
}

void applyVonNeumannBC(Array& var, const Constants& c) {
  for (int i=0; i<c.nx; ++i) {
    var(i, -1) = var(i, 1);
    var(i, c.ny) = var(i, c.ny-2);
  }
  for(int j=0; j<c.ny; ++j) {
    var(-1, j) = var(1, j);
    var(c.nx, j) = var(c.nx-2, j);
  }
}

void applyVxBC(Array& vx, const Constants &c) {
  applyNoSlipBC(vx, c);
  // Driven cavity flow
  for (int i=0; i<c.nx; ++i) {
    vx(i, c.ny) = 1;
  }
}

void applyVyBC(Array& vy, const Constants &c) {
  applyNoSlipBC(vy, c);
}

void applyBoundaryConditions(Variables& vars, const Constants& c) {
  applyVxBC(vars.vx, c);
  applyVyBC(vars.vy, c);
}

void render(const Variables& vars) {
  vars.vx.render();
}

int clamp(const int i, const int upper, const int lower) {
  return std::max(std::min(i, upper), lower);
}

real calcAdvection(const Array& f, const int i, const int j, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng, const Array& vx, const Array& vy) {
  // figure out where the current piece has come from (in index space)
  real oldi = i - dt*vx(i,j)/dx;
  real oldj = j - dt*vy(i,j)/dy;
  // clamp to int indices and ensure inside domain
  int ip = clamp(int(oldi+1),nx-1+ng,-ng);
  int im = clamp(int(oldi)  ,nx-1+ng,-ng);
  int jp = clamp(int(oldj+1),ny-1+ng,-ng);
  int jm = clamp(int(oldj)  ,ny-1+ng,-ng);
  // bilinearly interpolate
  return (f(ip, jp) + f(ip, jm) + f(im, jp) + f(im, jm))/4;
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

  setInitialConditions(vars, c);
  applyBoundaryConditions(vars, c);

  Array temp1(c, "temp1");
  Array temp2(c, "temp2");
  Array divw(c, "divw");

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
    f(i,j) = (vars.vx(i+1,j) - vars.vx(i-1,j))/(2*c.dx) + (vars.vy(i,j+1) - vars.vy(i,j-1))/(2*c.dy);
  };

  auto advectionKernel = [&](const Array& q, const int i, const int j) {
    return calcAdvection(q, i, j, c.dx, c.dy, c.dt, c.nx, c.ny, c.ng, vars.vx, vars.vy);
  };

  auto explicitAdvectionKernel = [&](const Array& q, const int i, const int j) {
    return vars.vx(i,j) * (q(i+1,j)-q(i-1,j))/(2*c.dx) + vars.vy(i,j) * (q(i,j+1) - q(i,j-1))/(2*c.dy);
  };

  auto eulerKernel = [&](Array& f, const Array& in, const int i, const int j) {
    f(i,j) += in(i,j)*c.dt;
  };

  auto vxProjectKernel = [&](Array& f, const Array& in, const int i, const int j) {
    f(i,j) -= (in(i+1,j)-in(i-1,j))/(2*c.dx);
  };

  auto vyProjectKernel = [&](Array& f, const Array& in, const int i, const int j) {
    f(i,j) -= (in(i,j+1)-in(i,j-1))/(2*c.dy);
  };

  HDFFile icFile("000000.hdf5");
  vars.vx.saveTo(icFile.file);
  vars.vy.saveTo(icFile.file);
  icFile.close();

  real t=0;
  while (t < c.totalTime) {
    // ADVECTION
    // implicit
    vars.vx.applyKernel(advectionKernel, temp1);
    vars.vy.applyKernel(advectionKernel, temp2);
    vars.vx.swapData(temp1);
    vars.vy.swapData(temp2);
    // explicit
    //vars.vx.applyKernel(explicitAdvectionKernel, temp1);
    //vars.vy.applyKernel(explicitAdvectionKernel, temp2);
    //vars.vx.applyKernel(eulerKernel, temp1);
    //vars.vy.applyKernel(eulerKernel, temp2);
    applyBoundaryConditions(vars, c);

    // DIFFUSION
    temp1.initialise(0);
    applyVxBC(temp1, c);
    applyVxBC(temp2, c);
    runJacobiIteration(temp1, temp2, vxDiffusionJacobiKernel);
    vars.vx.swapData(temp1);

    temp1.initialise(0);
    applyVyBC(temp1, c);
    applyVyBC(temp2, c);
    runJacobiIteration(temp1, temp2, vyDiffusionJacobiKernel);
    vars.vy.swapData(temp1);
    applyBoundaryConditions(vars, c);

    // PROJECTION
    divw.applyKernel(divergenceKernel);
    temp1.initialise(0);
    runJacobiIteration(temp1, temp2, projectionJacobiKernel);
    vars.p.swapData(temp1);
    applyVonNeumannBC(vars.p, c);
    vars.vx.applyKernel(vxProjectKernel, vars.p);
    vars.vy.applyKernel(vyProjectKernel, vars.p);
    applyBoundaryConditions(vars, c);

    t += c.dt;
  }

  //vars.vx.render();
  //vars.vy.render();

  HDFFile laterFile("000001.hdf5");
  vars.vx.saveTo(laterFile.file);
  vars.vy.saveTo(laterFile.file);
  vars.p.saveTo(laterFile.file);
  divw.saveTo(laterFile.file);
  temp1.saveTo(laterFile.file);
  temp2.saveTo(laterFile.file);
  icFile.close();
}

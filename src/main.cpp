#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>

#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>

#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>
#include <hdffile.hpp>
#include <ocl_utility.hpp>

void setInitialConditions(Variables& vars) {
  for (int i=vars.vx.nx/4; i<3*vars.vx.nx/4; ++i) {
    for (int j=vars.vx.ny/4; j<3*vars.vx.ny/4; ++j) {
      vars.vx(i,j) = 0.0;
    }
  }
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

template<class T>
T clamp(const T i, const T upper, const T lower) {
  return std::max(std::min(i, upper), lower);
}

void calcAdvection(Array& out, const Array& f, const int i, const int j, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng, const Array& vx, const Array& vy) {
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
  out(i,j) = fAv;
}

real calcJacobiStep(const Array& f, const real alpha, const real beta, const Array& b, const int i, const int j) {
  return (alpha*b(i,j) + f(i,j+1) + f(i,j-1) + f(i+1,j) + f(i-1,j))/beta;
}

void applyJacobiStep(Array& out, const Array& f, const real alpha, const real beta, const Array& b) {
#pragma omp parallel for collapse(2)
  for (int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = calcJacobiStep(f, alpha, beta, b, i, j);
    }
  }
}

void runJacobiIteration(Array& out, Array& in, const real alpha, const real beta, const Array& b, const int iterations=20) {
  for(int i=0; i<iterations; ++i) {
    applyJacobiStep(out, in, alpha, beta, b);
    in.swap(out);
  }
}

real ddx(const Array& f, const real dx, const int i, const int j) {
  return (f(i+1,j)-f(i-1,j))/(2.0f*dx);
}

real ddy(const Array& f, const real dy, const int i, const int j) {
  return (f(i,j+1) - f(i,j-1))/(2.0f*dy);
}

void advectImplicit(Array& out, const Array& f, const Array& vx, const Array& vy, const real dx, const real dy, const real dt, const int nx, const int ny, const int ng) {
  for (int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      calcAdvection(out, f, i, j, dx, dy, dt, nx, ny, ng, vx, vy);
    }
  }
}

void calcAdvectionTerm(Array& out, const Array& f, const Array& vx, const Array& vy, const real dx, const real dy) {
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = -(vx(i,j) * ddx(f, dx, i, j) + vy(i,j) * ddy(f, dy, i, j));
    }
  }
}

void advanceEuler(Array& out, const Array& ddt, const real dt) {
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) += ddt(i,j)*dt;
    }
  }
}

void calcDivergence(Array& out, const Array& fx, const Array& fy, const real dx, const real dy) {
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) = (fx(i,j) - fx(i-1,j))/dx + (fy(i,j) - fy(i,j-1))/dy;
    }
  }
}

void applyProjectionX(Array& out, const Array& f, const real dx) {
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) -= (f(i+1,j)-f(i,j))/dx;
    }
  }
}

void applyProjectionY(Array& out, const Array& f, const real dy) {
  for(int i=0; i<out.nx; ++i) {
    for(int j=0; j<out.ny; ++j) {
      out(i,j) -= (f(i,j+1)-f(i,j))/dy;
    }
  }
}

void runCPU() {
  const Constants c;

  Variables vars(c);

  setInitialConditions(vars);
  applyBoundaryConditions(vars);

  // Working arrays located at boundaries
  Array boundTemp1(c.nx, c.ny, c.ng, "boundTemp1");
  Array boundTemp2(c.nx, c.ny, c.ng, "boundTemp2");
  // Working arrays located at cell centres
  Array cellTemp1(c.nx+1, c.ny+1, c.ng, "cellTemp1");
  Array cellTemp2(c.nx+1, c.ny+1, c.ng, "cellTemp2");
  // Working array for divergence
  Array divw(c.nx+1, c.ny+1, c.ng, "divw");

  HDFFile icFile("000000.hdf5");
  vars.vx.saveTo(icFile.file);
  vars.vy.saveTo(icFile.file);
  icFile.close();

  real t=0;
  while (t < c.totalTime) {
    // ADVECTION
    // implicit
    advectImplicit(boundTemp1, vars.vx, vars.vx, vars.vy, c.dx, c.dy, c.dt, c.nx, c.ny, c.ng);
    advectImplicit(boundTemp2, vars.vy, vars.vx, vars.vy, c.dx, c.dy, c.dt, c.nx, c.ny, c.ng);
    vars.vx.swapData(boundTemp1);
    vars.vy.swapData(boundTemp2);
    // explicit
    //calcAdvectionTerm(boundTemp1, vars.vx, vars.vx, vars.vy, c.dx, c.dy);
    //calcAdvectionTerm(boundTemp2, vars.vy, vars.vx, vars.vy, c.dx, c.dy);
    //advanceEuler(vars.vx, boundTemp1, c.dt);
    //advanceEuler(vars.vy, boundTemp2, c.dt);

    applyBoundaryConditions(vars);

    // DIFFUSION
    real alpha = (c.dx*c.dy)/(c.nu*c.dt);
    real beta = 4+(c.dx*c.dy)/(c.nu*c.dt);

    // Diffuse vx
    // Implicit
    const real initialGuess = 0.0f;
    boundTemp1.initialise(initialGuess);
    applyVxBC(boundTemp1);
    applyVxBC(boundTemp2);
    runJacobiIteration(boundTemp2, boundTemp1, alpha, beta, vars.vx);
    vars.vx.swapData(boundTemp1);

    // Diffuse vy
    // Implicit
    const real initialGuess = 0.0f;
    boundTemp1.initialise(initialGuess);
    applyVyBC(boundTemp1);
    applyVyBC(boundTemp2);
    runJacobiIteration(boundTemp2, boundTemp1, alpha, beta, vars.vy);
    vars.vy.swapData(boundTemp1);

    applyBoundaryConditions(vars);

    // PROJECTION
    // Calculate divergence
    calcDivergence(divw, vars.vx, vars.vy, c.dx, c.dy);
    // Solve Poisson eq for pressure $\nabla^2 p = - \nabla \cdot v$
    cellTemp1.initialise(0);
    runJacobiIteration(cellTemp2, cellTemp1, -c.dx*c.dy, 4.0f, divw);
    vars.p.swapData(cellTemp1);
    applyVonNeumannBC(vars.p);
    // Project onto incompressible velocity space
    applyProjectionX(vars.vx, vars.p, c.dx);
    applyProjectionY(vars.vy, vars.p, c.dy);
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

int main() {
  runCPU();

  //int error = setDefaultPlatform("CUDA");
  //if (error < 0) return -1;

  //cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      //cl::Context::getDefault(), cl::Device::getDefault());

  //cl::Buffer d_A = cl::Buffer(h_A.begin(), h_A.end(), true);
  //cl::Buffer d_B = cl::Buffer(h_B.begin(), h_B.end(), true);
  //cl::Buffer d_C = cl::Buffer(CL_MEM_READ_WRITE, sizeof(float)*h_C.size());

  //// Allocate temp space local to workgroups
  //cl::LocalSpaceArg d_wrk = cl::Local(sizeof(float)*N);

  //cl::Program program = buildProgram("mat_mult.cl");
  //// Build kernel with extra workgroup space
  //auto mat_mult_cl = cl::KernelFunctor<
    //int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg
    //>(program, kernelName);

  //auto start = high_resolution_clock::now();

  //// Pass in workgroup
  //mat_mult_cl(cl::EnqueueArgs(cl::NDRange(c.ng, c.ng), cl::NDRange(N), cl::NDRange(N/4)), N, d_A, d_B, d_C, d_wrk);

  //cl::copy(d_C, h_C.begin(), h_C.end());
}

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
#include <openmp_kernels.hpp>

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
    real alpha = c.Re*c.dx*c.dy/c.dt;
    real beta = 4+alpha;

    // Diffuse vx
    // Implicit
    real initialGuess = 0.0f;
    boundTemp1.initialise(initialGuess);
    applyVxBC(boundTemp1);
    applyVxBC(boundTemp2);
    runJacobiIteration(boundTemp2, boundTemp1, alpha, beta, vars.vx);
    vars.vx.swapData(boundTemp1);
    // Explicit
    //calcDiffusionTerm(boundTemp1, vars.vx, c.dx, c.dy);
    //advanceEuler(vars.vx, boundTemp1, c.dt);

    // Diffuse vy
    // Implicit
    initialGuess = 0.0f;
    boundTemp1.initialise(initialGuess);
    applyVyBC(boundTemp1);
    applyVyBC(boundTemp2);
    runJacobiIteration(boundTemp2, boundTemp1, alpha, beta, vars.vy);
    vars.vy.swapData(boundTemp1);
    // Explicit
    //calcDiffusionTerm(boundTemp1, vars.vy, c.dx, c.dy);
    //advanceEuler(vars.vy, boundTemp1, c.dt);

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

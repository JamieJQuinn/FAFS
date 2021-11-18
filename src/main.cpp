#include <iostream>
#include <memory>
#include <cassert>

#include <ocl_utility.hpp>
#include <constants.hpp>
#include <precision.hpp>
#include <openmp_implementation.hpp>
#include <variables.hpp>
#include <array2d.hpp>
#include <opencl_kernels.hpp>

int runOCL() {
  const Constants c;

  int error = setDefaultPlatform("CUDA");
  if (error < 0) return -1;

  cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      cl::Context::getDefault(), cl::Device::getDefault());

  Variables vars(c);

  setInitialConditions(vars);
  // applyBoundaryConditions(vars); TODO wrap boundary conditions

  Kernels kernels;

  // TODO - put working arrays on device
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
    //advectImplicit(boundTemp1, vars.vx, vars.vx, vars.vy, c.dx, c.dy, c.dt, c.nx, c.ny, c.ng);
    //advectImplicit(boundTemp2, vars.vy, vars.vx, vars.vy, c.dx, c.dy, c.dt, c.nx, c.ny, c.ng);
    // DONE - implement swap of device pointers
    //vars.vx.swapData(boundTemp1);
    //vars.vy.swapData(boundTemp2);
    // explicit
    // DONE - implement explicit kernel
    //calcAdvectionTerm(boundTemp1, vars.vx, vars.vx, vars.vy, c.dx, c.dy);
    //calcAdvectionTerm(boundTemp2, vars.vy, vars.vx, vars.vy, c.dx, c.dy);
    // DONE - implement euler kernel
    //advanceEuler(vars.vx, boundTemp1, c.dt);
    //advanceEuler(vars.vy, boundTemp2, c.dt);

    //applyBoundaryConditions(vars);

    // DIFFUSION
    real alpha = c.Re*c.dx*c.dy/c.dt;
    real beta = 4+alpha;

    // Diffuse vx
    // Implicit
    // Done switch to fill kernel
    boundTemp1.fill(0.0f);
    //boundTemp1.initialise(initialGuess);
    // TODO wrap these
    //applyVxBC(boundTemp1);
    //applyVxBC(boundTemp2);
    // TODO write Jacobi iteration + wrap caller
    //runJacobiIteration(boundTemp2, boundTemp1, alpha, beta, vars.vx);
    //vars.vx.swapData(boundTemp1);
    // Explicit
    // DONE implement explicit diffusion kernel
    //calcDiffusionTerm(boundTemp1, vars.vx, c.dx, c.dy);
    //advanceEuler(vars.vx, boundTemp1, c.dt);

    // Diffuse vy TODO do same to this
    // Implicit
    //initialGuess = 0.0f;
    //boundTemp1.initialise(initialGuess);
    //applyVyBC(boundTemp1);
    //applyVyBC(boundTemp2);
    //runJacobiIteration(boundTemp2, boundTemp1, alpha, beta, vars.vy);
    //vars.vy.swapData(boundTemp1);
    // Explicit
    //calcDiffusionTerm(boundTemp1, vars.vy, c.dx, c.dy);
    //advanceEuler(vars.vy, boundTemp1, c.dt);

    //applyBoundaryConditions(vars);

    // PROJECTION
    // Calculate divergence TODO implement kernel
    //calcDivergence(divw, vars.vx, vars.vy, c.dx, c.dy);
    // Solve Poisson eq for pressure $\nabla^2 p = - \nabla \cdot v$
    //cellTemp1.initialise(0); // TODO fill
    //runJacobiIteration(cellTemp2, cellTemp1, -c.dx*c.dy, 4.0f, divw);
    //vars.p.swapData(cellTemp1);
    //applyVonNeumannBC(vars.p);
    // Project onto incompressible velocity space
    //applyProjectionX(vars.vx, vars.p, c.dx); // TODO implement this
    //applyProjectionY(vars.vy, vars.p, c.dy);
    //applyBoundaryConditions(vars);

    t += c.dt;
  }

  HDFFile laterFile("000001.hdf5");
  // TODO implement fetching of data when writing
  //vars.vx.saveTo(laterFile.file);
  //vars.vy.saveTo(laterFile.file);
  //vars.p.saveTo(laterFile.file);
  //divw.saveTo(laterFile.file);
  icFile.close();



  return 0;
}

int main() {
  return runOCL();
}

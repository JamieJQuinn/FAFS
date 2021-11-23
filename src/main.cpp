#include <iostream>
#include <memory>
#include <cassert>

#include <ocl_utility.hpp>
#include <constants.hpp>
#include <precision.hpp>
#include <openmp_implementation.hpp>
#include <variables.hpp>
#include <array2d.hpp>
#include <ocl_array.hpp>
#include <user_kernels.hpp>
#include <hdffile.hpp>

void applyVxBC(OpenCLArray& vx) {
  applyNoSlipBC(vx);
  vx.setUpperBoundary(1.0f);
}

void applyVyBC(OpenCLArray& vy) {
  applyNoSlipBC(vy);
}

void applyPressureBC(OpenCLArray& p) {
  applyVonNeumannBC(p);
}

void applyBoundaryConditions(Variables<OpenCLArray>& vars) {
  applyVxBC(vars.vx);
  applyVyBC(vars.vy);
  //applyPressureBC(vars.p);
}

void setInitialConditions(Variables<OpenCLArray>& vars) {
  vars.vx.fill(0.0f, true);
  vars.vy.fill(0.0f, true);
  vars.p.fill(0.0f, true);
}

int runOCL() {
  const Constants c;

  c.print();

  int error = setDefaultPlatform("CUDA");
  if (error < 0) return -1;

  cl::DeviceCommandQueue deviceQueue = cl::DeviceCommandQueue::makeDefault(
      cl::Context::getDefault(), cl::Device::getDefault());

  Variables <OpenCLArray> vars(c);

  setInitialConditions(vars);
  applyBoundaryConditions(vars);

  // Working arrays located at boundaries
  OpenCLArray boundTemp1(c.nx, c.ny, c.ng, "boundTemp1");
  OpenCLArray boundTemp2(c.nx, c.ny, c.ng, "boundTemp2");
  // Working arrays located at cell centres
  OpenCLArray cellTemp1(c.nx+1, c.ny+1, c.ng, "cellTemp1");
  OpenCLArray cellTemp2(c.nx+1, c.ny+1, c.ng, "cellTemp2");
  // Working array for divergence
  OpenCLArray divw(c.nx+1, c.ny+1, c.ng, "divw");

  HDFFile icFile("000000.hdf5", false);
  vars.vx.saveTo(icFile.file);
  vars.vy.saveTo(icFile.file);
  icFile.close();

  real t=0;
  while (t < c.totalTime) {
    // ADVECTION
    if(c.isAdvectionImplicit) {
      advectImplicit(boundTemp1, vars.vx, vars.vx, vars.vy, c.dx, c.dy, c.dt);
      advectImplicit(boundTemp2, vars.vy, vars.vx, vars.vy, c.dx, c.dy, c.dt);
      vars.vx.swapData(boundTemp1);
      vars.vy.swapData(boundTemp2);
    } else {
      calcAdvectionTerm(boundTemp1, vars.vx, vars.vx, vars.vy, c.dx, c.dy);
      calcAdvectionTerm(boundTemp2, vars.vy, vars.vx, vars.vy, c.dx, c.dy);
      advanceEuler(vars.vx, boundTemp1, c.dt);
      advanceEuler(vars.vy, boundTemp2, c.dt);
    }

    applyBoundaryConditions(vars);

    // DIFFUSION
    if(c.isDiffusionImplicit) {
      real alpha = c.Re*c.dx*c.dy/c.dt;
      real beta = 4.0f+alpha;

      boundTemp1.fill(0.0f, true);
      applyVxBC(boundTemp1);
      applyVxBC(boundTemp2);
      runJacobiIteration(vars.vx, boundTemp1, boundTemp2, alpha, beta, vars.vx);

      boundTemp1.fill(0.0f, true);
      applyVyBC(boundTemp1);
      applyVyBC(boundTemp2);
      runJacobiIteration(vars.vy, boundTemp1, boundTemp2, alpha, beta, vars.vy);
    } else {
      calcDiffusionTerm(boundTemp1, vars.vx, c.dx, c.dy, c.Re);
      advanceEuler(vars.vx, boundTemp1, c.dt);

      calcDiffusionTerm(boundTemp1, vars.vy, c.dx, c.dy, c.Re);
      advanceEuler(vars.vy, boundTemp1, c.dt);
    }

    applyBoundaryConditions(vars);

    // PROJECTION
    divw.fill(0.0f, true);
    calcDivergence(divw, vars.vx, vars.vy, c.dx, c.dy);
    // Solve Poisson eq for pressure $\nabla^2 p = - \nabla \cdot v$
    cellTemp1.fill(0.0f, true);
    runJacobiIteration(vars.p, cellTemp1, cellTemp2, -c.dx*c.dy, 4.0f, divw);
    applyVonNeumannBC(vars.p);
    // Project onto incompressible velocity space
    applyProjectionX(vars.vx, vars.p, c.dx);
    applyProjectionY(vars.vy, vars.p, c.dy);

    applyBoundaryConditions(vars);

    t += c.dt;
  }

  HDFFile laterFile("000001.hdf5", false);
  vars.vx.saveTo(laterFile.file);
  vars.vy.saveTo(laterFile.file);
  vars.p.saveTo(laterFile.file);
  divw.saveTo(laterFile.file);
  icFile.close();

  return 0;
}

int main() {
  //return runOCL();
  return runCPU();
}

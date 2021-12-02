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
  vx.setLeftBoundary(0.0f);
  vx.setRightBoundary(0.0f);
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
  applyPressureBC(vars.p);
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
  vars.p.saveTo(icFile.file);
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

    real dx2dy2 = c.dx*c.dx + c.dy*c.dy;

    // DIFFUSION
    if(c.isDiffusionImplicit) {
      real alpha = 1.0f/(1.0f + 2.0f*c.dt/c.Re*(1.0f/(c.dx*c.dx) + 1.0f/(c.dy*c.dy)));
      real beta  = -c.Re*c.dx*c.dx/c.dt;
      real gamma = -c.Re*c.dy*c.dy/c.dt;

      boundTemp1.fill(0.0f, true);
      applyVxBC(boundTemp1);
      applyVxBC(boundTemp2);
      runJacobiIteration(vars.vx, boundTemp1, boundTemp2, alpha, beta, gamma, vars.vx);

      boundTemp1.fill(0.0f, true);
      applyVyBC(boundTemp1);
      applyVyBC(boundTemp2);
      runJacobiIteration(vars.vy, boundTemp1, boundTemp2, alpha, beta, gamma, vars.vy);
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
    real alpha = -0.5f*(c.dx*c.dx*c.dy*c.dy)/dx2dy2;
    real beta = c.dx*c.dx;
    real gamma = c.dy*c.dy;
    cellTemp1.fill(0.0f, true);
    cellTemp2.fill(0.0f, true);
    for(int i=0; i<200; ++i) {
      applyPressureBC(cellTemp1);
      g_kernels.applyJacobiStep(cellTemp2.interior, cellTemp2.getDeviceData(), cellTemp1.getDeviceData(), alpha, beta, gamma, divw.getDeviceData(), cellTemp2.nx, cellTemp2.ny, cellTemp2.ng);
      cellTemp1.swapData(cellTemp2);
    }
    vars.p.swapData(cellTemp1);
    applyPressureBC(vars.p);
    // Project onto incompressible velocity space
    applyProjectionX(vars.vx, vars.p, c.dx);
    applyProjectionY(vars.vy, vars.p, c.dy);

    applyBoundaryConditions(vars);

    t += c.dt;
  }

  // DEBUG
  vars.p.toHost();

  Array cellTemp3(c.nx+1, c.ny+1, c.ng, "cellTemp3");
  for(int i=0; i<cellTemp3.nx; ++i) {
    for(int j=0; j<cellTemp3.ny; ++j) {
      cellTemp3(i,j) = (vars.p(i+1,j) + vars.p(i-1,j) + vars.p(i, j+1) + vars.p(i,j-1) - 4.0*vars.p(i,j))/(c.dx*c.dx);
    }
  }
  divw.toHost();
  for(int i=0; i<cellTemp3.nx; ++i) {
    for(int j=0; j<cellTemp3.ny; ++j) {
      cellTemp3(i,j) -= divw(i,j);
    }
  }

  Array dpdx(c.nx, c.ny, c.ng, "dpdx");
  Array dpdy(c.nx, c.ny, c.ng, "dpdy");
  for(int i=0; i<dpdx.nx; ++i) {
    for(int j=0; j<dpdx.ny; ++j) {
      real dfdxjp = (vars.p(i+1,j+1) - vars.p(i,j+1))/c.dx;
      real dfdxj = (vars.p(i+1,j) - vars.p(i,j))/c.dx;
      real dfdx = (dfdxjp + dfdxj)/2.0;
      dpdx(i,j) = -dfdx;

      real dfdyip = (vars.p(i+1,j+1) - vars.p(i+1,j))/c.dy;
      real dfdyi = (vars.p(i,j+1) - vars.p(i,j))/c.dy;
      real dfdy = (dfdyip + dfdyi)/2.0;
      dpdy(i,j) = -dfdy;
    }
  }

  calcDivergence(divw, vars.vx, vars.vy, c.dx, c.dy);
  // END DEBUG

  HDFFile laterFile("000001.hdf5", false);
  vars.vx.saveTo(laterFile.file);
  vars.vy.saveTo(laterFile.file);
  vars.p.saveTo(laterFile.file);
  divw.saveTo(laterFile.file);
  // DEBUG
  cellTemp3.saveTo(laterFile.file);
  dpdx.saveTo(laterFile.file);
  dpdy.saveTo(laterFile.file);
  // END DEBUG
  icFile.close();

  return 0;
}

int main() {
  return runOCL();
  //return runCPU();
}

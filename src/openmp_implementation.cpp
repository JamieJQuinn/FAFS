#include <openmp_kernels.hpp>
#include <array2d.hpp>
#include <variables.hpp>
#include <precision.hpp>
#include <hdffile.hpp>
#include <openmp_kernels.hpp>


void setInitialConditions(Variables& vars) {
  for (int i=0; i<vars.vx.nx; ++i) {
    for (int j=0; j<vars.vx.ny; ++j) {
      vars.vx(i,j) = 0.0f;
      vars.vy(i,j) = 0.0f;
    }
  }
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
    real beta = 4.0f+alpha;

    // Diffuse vx
    // Implicit
    real initialGuess = 0.0f;
    boundTemp1.fill(initialGuess);
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
    boundTemp1.fill(initialGuess);
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
    cellTemp1.fill(0);
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


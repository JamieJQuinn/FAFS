#include <ocl_array.hpp>

void runJacobiIteration(OpenCLArray& out, OpenCLArray& initialGuess, OpenCLArray& temp, const real alpha, const real beta, const real gamma, const OpenCLArray& b, const int iterations) {
  for(int i=0; i<iterations; ++i) {
    g_kernels.applyJacobiStep(temp.interior, temp.getDeviceData(), initialGuess.getDeviceData(), alpha, beta, gamma, b.getDeviceData(), temp.nx, temp.ny, temp.ng);
    initialGuess.swapData(temp);
  }
  out.swapData(initialGuess);
}

void calcDiffusionTerm(OpenCLArray& out, const OpenCLArray& f, const real dx, const real dy, const real Re) {
  g_kernels.calcDiffusionTerm(out.interior, out.getDeviceData(), f.getDeviceData(), dx, dy, Re, out.nx, out.ny, out.ng);
}

void calcAdvectionTerm(OpenCLArray& out, const OpenCLArray& f, const OpenCLArray& vx, const OpenCLArray& vy, const real dx, const real dy) {
  g_kernels.calcAdvectionTerm(out.interior, out.getDeviceData(), f.getDeviceData(), vx.getDeviceData(), vy.getDeviceData(), dx, dy, out.nx, out.ny, out.ng);
}

void advanceEuler(OpenCLArray& out, const OpenCLArray& ddt, const real dt) {
  g_kernels.advanceEuler(out.interior, out.getDeviceData(), ddt.getDeviceData(), dt, out.nx, out.ny, out.ng);
}

void applyVonNeumannBC_y(OpenCLArray& out) {
  g_kernels.applyVonNeumannBC_y(out.lowerBound, out.getDeviceData(), out.nx, out.ny, out.ng);
}

void applyVonNeumannBC_x(OpenCLArray& out) {
  g_kernels.applyVonNeumannBC_x(out.leftBound, out.getDeviceData(), out.nx, out.ny, out.ng);
}

void applyVonNeumannBC(OpenCLArray& out) {
  applyVonNeumannBC_y(out);
  applyVonNeumannBC_x(out);
}

void applyNoSlipBC(OpenCLArray& var) {
  var.setUpperBoundary(0.0f);
  var.setLowerBoundary(0.0f);
  var.setLeftBoundary(0.0f);
  var.setRightBoundary(0.0f);
}

void calcDivergence(OpenCLArray& out, OpenCLArray& fx, OpenCLArray& fy, const real dx, const real dy) {
  g_kernels.calcDivergence(out.interior, out.getDeviceData(), fx.getDeviceData(), fy.getDeviceData(), dx, dy, out.nx, out.ny, out.ng);
}

void applyProjectionX(OpenCLArray& out, OpenCLArray& f, const real dx) {
  g_kernels.applyProjectionX(out.interior, out.getDeviceData(), f.getDeviceData(), dx, out.nx, out.ny, out.ng);
}

void applyProjectionY(OpenCLArray& out, OpenCLArray& f, const real dy) {
  g_kernels.applyProjectionY(out.interior, out.getDeviceData(), f.getDeviceData(), dy, out.nx, out.ny, out.ng);
}

void advectImplicit(OpenCLArray& out, OpenCLArray& f, OpenCLArray& vx, OpenCLArray& vy, const real dx, const real dy, const real dt) {
  g_kernels.advect(out.interior, out.getDeviceData(), f.getDeviceData(), vx.getDeviceData(), vy.getDeviceData(), dx, dy, dt, out.nx, out.ny, out.ng);
}

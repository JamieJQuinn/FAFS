#include <ocl_array.hpp>

void runJacobiIteration(OpenCLArray& out, OpenCLArray& in, const real alpha, const real beta, const OpenCLArray& b, const int iterations) {
  for(int i=0; i<iterations; ++i) {
    g_kernels.applyJacobiStep(out.interior, out.getDeviceData(), in.getDeviceData(), alpha, beta, b.getDeviceData(), out.nx, out.ny, out.ng);
    in.swap(out);
  }
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
